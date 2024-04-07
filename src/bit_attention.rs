use crate::{
    bit_linear::{Bitlinear, BitlinearCfg},
    utils_tensor::scaled_dot_product_attention,
};
use anyhow::{anyhow, Ok, Result};
use candle_core::{DType, Device, Tensor};
use candle_einops::einops;
use candle_nn::{rotary_emb::rope_i, VarBuilder};
use tracing::instrument;

#[derive(Debug, Clone, Copy)]
pub struct BitAttentionCfg {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub dropout: f32,
    pub eps: f64,
    pub max_seq_len: usize,
}

#[derive(Debug)]
pub struct BitAttention {
    q_proj: Bitlinear,
    k_proj: Bitlinear,
    v_proj: Bitlinear,
    o_proj: Bitlinear,
    dropout: f32,
    n_heads: usize,
    n_kv_heads: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    max_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((max_seq_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl BitAttention {
    pub fn load(cfg: BitAttentionCfg, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.dim / cfg.n_heads;
        if cfg.n_heads % cfg.n_kv_heads != 0 {
            return Err(anyhow!(
                "query_heads must be divisible by kv_heads, got: {} and {}",
                cfg.n_heads,
                cfg.n_kv_heads
            ));
        }
        if (cfg.dim % cfg.n_heads) != 0 {
            return Err(anyhow!(
                "dim must be divisible by query_heads, got: {} and {}",
                cfg.n_heads,
                cfg.dim
            ));
        }
        if (cfg.dim % cfg.n_heads) != 0 {
            return Err(anyhow!(
                "dim must be divisible by n_kv_heads, got: {} and {}",
                cfg.n_heads,
                cfg.dim
            ));
        }
        if head_dim % 8 != 0 {
            return Err(anyhow!(
                "head_dim must be divisible by 8, got: {}",
                head_dim
            ));
        }
        if head_dim > 128 {
            return Err(anyhow!("head_dim must be less than or equal to 128"));
        }

        let q_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: cfg.n_heads * head_dim,
                eps: cfg.eps,
            },
            vb.pp("q_proj"),
        )?;
        let k_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: cfg.n_kv_heads * head_dim,
                eps: cfg.eps,
            },
            vb.pp("k_proj"),
        )?;
        let v_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: cfg.n_kv_heads * head_dim,
                eps: cfg.eps,
            },
            vb.pp("v_proj"),
        )?;

        let o_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: cfg.dim,
                eps: cfg.eps,
            },
            vb.pp("o_proj"),
        )?;

        let (cos, sin) = precompute_freqs_cis(head_dim, 10000., cfg.max_seq_len, &vb.device())?;

        Ok(BitAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            dropout: cfg.dropout,
            cos,
            sin,
            kv_cache: None,
        })
    }

    #[instrument]
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let x = rope_i(&x.contiguous()?, &cos, &sin)?;
        let x = x.to_dtype(dtype)?;
        Ok(x)
    }

    #[instrument]
    pub fn forward(&mut self, x: &Tensor, is_causal: bool, index_pos: usize) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = einops!("b n ({self.n_heads} d) -> b n {self.n_heads} d", q);
        let k = einops!("b n ({self.n_kv_heads} d) -> b n {self.n_kv_heads} d", k);
        let v = einops!("b n ({self.n_kv_heads} d) -> b n {self.n_kv_heads} d", v);

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let scale = (q.dims4()?.3 as f64).sqrt();
        let x = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None,
            Some(self.dropout),
            Some(is_causal),
            Some(scale),
        )?;

        let x = einops!("b n h d -> b n (h d)", x);
        let x = self.o_proj.forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bit_attention_tests {
    use crate::{
        bit_attention::{BitAttention, BitAttentionCfg},
        utils_tensor::device,
    };
    use candle_core::{DType, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn forward_produces_expected_shape_f32() -> anyhow::Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let input_tensor = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &device)?;
        let mut bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                dropout: 0.1,
                eps: 1e-6,
                max_seq_len: 64,
            },
            vb,
        )?;

        let output_tensor = bit_attention.forward(&input_tensor, true, 0).unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }
}
