use crate::{
    bit_linear::Bitlinear,
    utils_tensor::{scaled_dot_product_gqa, ScaledDotProductCfg},
};
use anyhow::{anyhow, Result};
use candle_core::Tensor;
use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder};

pub struct BitAttentionCfg {
    pub embed_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub dropout: f32,
    pub layer_norm_enabled: bool,
    pub layer_norm_eps: f64,
    pub bit_attention_eps: f64,
}

pub struct BitAttention {
    q_proj: Bitlinear,
    k_proj: Bitlinear,
    v_proj: Bitlinear,
    norm: Option<candle_nn::LayerNorm>,
    out_proj: Bitlinear,
    dropout: f32,
    query_heads: usize,
    kv_heads: usize,
}

impl BitAttention {
    pub fn load(cfg: BitAttentionCfg, vb: VarBuilder) -> Result<Self> {
        let kv_embed_dim = cfg.embed_dim / cfg.query_heads * cfg.kv_heads;
        let head_dim = cfg.embed_dim / cfg.query_heads;
        if cfg.query_heads % cfg.kv_heads != 0 {
            return Err(anyhow!("query_heads must be divisible by kv_heads"));
        }
        if (cfg.embed_dim % cfg.query_heads) != 0 {
            return Err(anyhow!("embed_dim must be divisible by query_heads"));
        }
        if (cfg.embed_dim % cfg.kv_heads) != 0 {
            return Err(anyhow!("embed_dim must be divisible by kv_heads"));
        }
        if head_dim % 8 != 0 {
            return Err(anyhow!("head_dim must be divisible by 8"));
        }
        if head_dim > 128 {
            return Err(anyhow!("head_dim must be less than or equal to 128"));
        }

        let q_proj = Bitlinear::load(cfg.embed_dim, cfg.embed_dim, 1, vb.pp("q_proj"))?;
        let k_proj = Bitlinear::load(cfg.embed_dim, kv_embed_dim, 1, vb.pp("k_proj"))?;
        let v_proj = Bitlinear::load(cfg.embed_dim, kv_embed_dim, 1, vb.pp("v_proj"))?;

        let norm = match cfg.layer_norm_enabled {
            true => {
                let config = LayerNormConfig {
                    eps: cfg.layer_norm_eps,
                    ..LayerNormConfig::default()
                };
                Some(layer_norm(kv_embed_dim, config, vb.pp("norm"))?)
            }
            false => None,
        };

        let out_proj = Bitlinear::load(kv_embed_dim, cfg.embed_dim, 1, vb.pp("out_proj"))?;

        Ok(BitAttention {
            q_proj,
            k_proj,
            v_proj,
            norm,
            query_heads: cfg.query_heads,
            kv_heads: cfg.kv_heads,
            out_proj,
            dropout: cfg.dropout,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool,
        is_causal: bool,
        average_attn_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // shape (b, n, d)
        let q = self.q_proj.forward(&query)?;
        let k = self.k_proj.forward(&key)?;
        let v = self.v_proj.forward(&value)?;

        // NOTE: original library uses einops to do this, we have to implement it ourselves
        // need to replicate "b n (h d) -> b n h d"
        let q = {
            let (batch_size, num_queries, total_depth) = q.dims3()?;
            let depth_per_head = total_depth / self.query_heads;
            q.reshape((batch_size, num_queries, self.query_heads, depth_per_head))?
        };
        let k = {
            let (batch_size, num_queries, total_depth) = k.dims3()?;
            let depth_per_head = total_depth / self.kv_heads;
            k.reshape((batch_size, num_queries, self.kv_heads, depth_per_head))?
        };
        let v = {
            let (batch_size, num_queries, total_depth) = v.dims3()?;
            let depth_per_head = total_depth / self.kv_heads;
            v.reshape((batch_size, num_queries, self.kv_heads, depth_per_head))?
        };

        let (x, attn_weights) = scaled_dot_product_gqa(
            q,
            k,
            v,
            ScaledDotProductCfg {
                is_causal,
                need_weights,
                average_attn_weights,
                force_grouped: false,
                dropout: self.dropout,
            },
        )?;

        // x = rearrange(x, "b n h d -> b n (h d)")
        let x_dims = x.dims4()?;
        let x = x.reshape((x_dims.0, x_dims.1, x_dims.2 * x_dims.3))?;

        // Original source mentions the magneto paper, need to read on this and the impact
        let x = match self.norm {
            Some(ref norm) => norm.forward(&x)?,
            None => x,
        };

        // Linear projection on the attn outputs
        let x = self.out_proj.forward(&x)?;

        Ok((x, attn_weights))
    }
}

#[cfg(test)]
mod bit_attention_tests {
    use crate::{
        bit_attention::{BitAttention, BitAttentionCfg},
        utils_tensor::device,
    };
    use candle_core::{safetensors, Result, Tensor};
    use candle_nn::VarBuilder;
    use test::Bencher;

    const DEFAULT_CFG: BitAttentionCfg = BitAttentionCfg {
        embed_dim: 512,
        kv_heads: 8,
        query_heads: 4,
        dropout: 0.1,
        layer_norm_enabled: false,
        bit_attention_eps: 1e-5,
        layer_norm_eps: 1e-5,
    };

    #[test]
    fn it_matches_python_snapshot() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let safetensor =
            safetensors::load("src/test_data/bit_attention_test.safetensors", &device)?;

        let input_tensor = safetensor.get("input_small").unwrap();
        let expected_output_tensor = safetensor.get("output_small").unwrap();
        let expected_attn_weights = safetensor.get("attn_weights_small").unwrap();

        let bit_attention = BitAttention::load(DEFAULT_CFG, vb).unwrap();

        let (output_tensor, attn_weights) = bit_attention
            .forward(
                input_tensor.clone(),
                input_tensor.clone(),
                input_tensor.clone(),
                true,
                false,
                false,
            )
            .unwrap();

        assert_eq!(output_tensor.shape(), expected_output_tensor.shape());
        assert_eq!(attn_weights.unwrap().shape(), expected_attn_weights.shape());

        Ok(())
    }

    #[bench]
    fn bench_bit_attention(b: &mut Bencher) -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let input = Tensor::randn(0.0f32, 1.0f32, (1, 10, 128), &device)?;
        let bit_attention = BitAttention::load(DEFAULT_CFG, vb).unwrap();

        b.iter(|| {
            for _ in 1..100 {
                bit_attention
                    .forward(
                        input.clone(),
                        input.clone(),
                        input.clone(),
                        true,
                        false,
                        false,
                    )
                    .unwrap();
            }
        });

        Ok(())
    }
}
