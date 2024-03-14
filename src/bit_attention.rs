use crate::{bit_linear::Bitlinear, utils_tensor::scaled_dot_product_attention};
use anyhow::{anyhow, Result};
use candle_core::Tensor;
use candle_einops::einops;
use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder};

pub struct BitAttentionCfg {
    pub embed_dim: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub dropout: f32,
    pub layer_norm_enabled: bool,
    pub eps: f32,
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
    span: tracing::Span,
}

impl BitAttention {
    pub fn load(cfg: BitAttentionCfg, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "bit-attention");
        let kv_embed_dim = cfg.embed_dim / cfg.query_heads * cfg.kv_heads;
        let head_dim = cfg.embed_dim / cfg.query_heads;
        if cfg.query_heads % cfg.kv_heads != 0 {
            return Err(anyhow!(
                "query_heads must be divisible by kv_heads, got: {} and {}",
                cfg.query_heads,
                cfg.kv_heads
            ));
        }
        if (cfg.embed_dim % cfg.query_heads) != 0 {
            return Err(anyhow!(
                "embed_dim must be divisible by query_heads, got: {} and {}",
                cfg.query_heads,
                cfg.embed_dim
            ));
        }
        if (cfg.embed_dim % cfg.kv_heads) != 0 {
            return Err(anyhow!(
                "embed_dim must be divisible by kv_heads, got: {} and {}",
                cfg.kv_heads,
                cfg.embed_dim
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
            cfg.embed_dim,
            cfg.embed_dim,
            1,
            8,
            cfg.eps,
            true,
            vb.pp("q_proj"),
        )?;
        let k_proj = Bitlinear::load(
            cfg.embed_dim,
            kv_embed_dim,
            1,
            8,
            cfg.eps,
            true,
            vb.pp("k_proj"),
        )?;
        let v_proj = Bitlinear::load(
            cfg.embed_dim,
            kv_embed_dim,
            1,
            8,
            cfg.eps,
            true,
            vb.pp("v_proj"),
        )?;

        let norm = match cfg.layer_norm_enabled {
            true => {
                let config = LayerNormConfig {
                    eps: cfg.eps.into(),
                    ..LayerNormConfig::default()
                };
                Some(layer_norm(kv_embed_dim, config, vb.pp("norm"))?)
            }
            false => None,
        };

        let out_proj = Bitlinear::load(
            kv_embed_dim,
            cfg.embed_dim,
            1,
            8,
            cfg.eps,
            true,
            vb.pp("out_proj"),
        )?;

        Ok(BitAttention {
            span,
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
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        is_causal: bool,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();

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

        let scale = (*query.dims().last().unwrap() as f64).sqrt();
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
        let x = match self.norm {
            Some(ref norm) => norm.forward(&x)?,
            None => x,
        };

        let x = self.out_proj.forward(&x)?;

        Ok(x)
    }
}

#[cfg(test)]
mod bit_attention_tests {
    use crate::{
        bit_attention::{BitAttention, BitAttentionCfg},
        utils_tensor::device,
    };
    use candle_core::{Result, Tensor};
    use candle_nn::VarBuilder;

    const DEFAULT_CFG: BitAttentionCfg = BitAttentionCfg {
        embed_dim: 64,
        kv_heads: 8,
        query_heads: 8,
        dropout: 0.1,
        layer_norm_enabled: true,
        eps: 1e-6,
    };

    #[test]
    fn forward_produces_expected_shape() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);

        let input_tensor = Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &device)?;
        let bit_attention = BitAttention::load(DEFAULT_CFG, vb).unwrap();

        let output_tensor = bit_attention
            .forward(&input_tensor, &input_tensor, &input_tensor, true)
            .unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }
}
