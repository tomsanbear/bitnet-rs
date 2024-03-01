use crate::config::Config;
use crate::rotary_embedding::RotaryEmbedding;
/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{linear_no_bias, Linear};
use std::sync::Arc;

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_flash_attn: bool,
}

impl Attention {
    pub fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    fn repeat_kv(&self, xs: Tensor) -> Result<Tensor> {
        let n_rep = self.num_kv_groups;
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
            xs.unsqueeze(2)?
                .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
                .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
        }
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = x.dims3()?;

        let query_states = self.q_proj.forward(x)?;
        let key_states = self.k_proj.forward(x)?;
        let value_states = self.v_proj.forward(x)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        println!("query_states: {:?}", query_states.dims());
        println!("key_states: {:?}", key_states.dims());
        println!("value_states: {:?}", value_states.dims());

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, 0)?; // seqlen_offset = 0 , TODO: fix and figure out how it works

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = self.repeat_kv(key_states)?;
        let value_states = self.repeat_kv(value_states)?;

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = query_states.transpose(1, 2)?;
            let k = key_states.transpose(1, 2)?;
            let v = value_states.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[cfg(test)]
mod attention_tests {
    use candle_core::{Device, Result, Tensor};
    use candle_nn::{Activation, VarBuilder};

    use super::Attention;
    use crate::config::Config;
    use crate::rotary_embedding::RotaryEmbedding;
    use std::sync::Arc;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let config = Config {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: 4096,
            use_flash_attn: false,
        };
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), &config, vb.device())?;
        let t = Attention::new(Arc::new(rotary_emb), &config, vb)?;

        assert_eq!(t.num_heads, 32);
        assert_eq!(t.num_kv_heads, 8);
        assert_eq!(t.num_kv_groups, 4);
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.hidden_size, 4096);

        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let config = Config {
            vocab_size: 32000,
            hidden_size: 1024,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: 1024,
            use_flash_attn: false,
        };
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), &config, vb.device())?;
        let mut t = Attention::new(Arc::new(rotary_emb), &config, vb)?;

        let x = Tensor::randn(0f64, 1f64, (1, config.hidden_size), &Device::Cpu)?.unsqueeze(0)?;
        let output = t.forward(&x)?;

        assert_eq!(output.dims(), &[1, 1, 1024]);

        Ok(())
    }
}
