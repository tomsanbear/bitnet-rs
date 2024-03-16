use crate::{
    bit_linear::{Bitlinear, BitlinearCfg},
    utils_tensor::scaled_dot_product_attention,
};
use anyhow::{anyhow, Result};
use candle_core::{Tensor, D};
use candle_einops::einops;
use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder};
use tracing::span;

pub struct BitAttentionCfg {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub dropout: f32,
    pub bias: bool,
    pub layer_norm_enabled: bool,
    pub eps: f32,
}

pub struct BitAttention {
    qkv_proj: Bitlinear,
    o_proj: Bitlinear,
    norm: Option<candle_nn::LayerNorm>,
    dropout: f32,
    dim: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    span: tracing::Span,
}

impl BitAttention {
    pub fn load(cfg: BitAttentionCfg, vb: VarBuilder) -> Result<Self> {
        let span = span!(tracing::Level::TRACE, "bit-attention");
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

        let total_head_dim = (cfg.n_heads + (2 * cfg.n_kv_heads)) * head_dim;
        let qkv_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: total_head_dim,
                num_groups: 1,
                b: 8,
                eps: cfg.eps,
                bias: true,
            },
            vb.pp("qkv_proj"),
        )?;

        let norm = match cfg.layer_norm_enabled {
            true => {
                let config = LayerNormConfig {
                    eps: cfg.eps.into(),
                    ..LayerNormConfig::default()
                };
                Some(layer_norm(
                    head_dim * cfg.n_kv_heads,
                    config,
                    vb.pp("layer_norm"),
                )?)
            }
            false => None,
        };

        let o_proj = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: cfg.dim,
                num_groups: 1,
                b: 8,
                eps: cfg.eps,
                bias: true,
            },
            vb.pp("o_proj"),
        )?;

        Ok(BitAttention {
            span,
            qkv_proj,
            o_proj,
            norm,
            dim: cfg.dim,
            head_dim,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            dropout: cfg.dropout,
        })
    }

    pub fn forward(&mut self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
        let _enter = self.span.enter();

        let qkv = self.qkv_proj.forward(x)?;

        let kv_size = self.n_kv_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, self.dim)?;
        let k = qkv.narrow(D::Minus1, self.dim, kv_size)?;
        let v = qkv.narrow(D::Minus1, self.dim + kv_size, kv_size)?;

        let q = einops!("b n ({self.n_heads} d) -> b n {self.n_heads} d", q);
        let k = einops!("b n ({self.n_kv_heads} d) -> b n {self.n_kv_heads} d", k);
        let v = einops!("b n ({self.n_kv_heads} d) -> b n {self.n_kv_heads} d", v);

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
        let x = match self.norm {
            Some(ref norm) => norm.forward(&x)?,
            None => x,
        };
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
                bias: true,
                dropout: 0.1,
                layer_norm_enabled: true,
                eps: 1e-6,
            },
            vb,
        )?;

        let output_tensor = bit_attention.forward(&input_tensor, true).unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }

    #[test]
    fn forward_produces_expected_shape_f16() -> anyhow::Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F16, &device);

        let input_tensor =
            Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &device)?.to_dtype(DType::F16)?;
        let mut bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                bias: true,
                dropout: 0.1,
                layer_norm_enabled: true,
                eps: 1e-6,
            },
            vb,
        )?;

        let output_tensor = bit_attention.forward(&input_tensor, true).unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }

    #[test]
    fn forward_produces_expected_shape_bf16() -> anyhow::Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::BF16, &device);

        let input_tensor =
            Tensor::randn(0.0f32, 1.0f32, (2, 8, 64), &device)?.to_dtype(DType::BF16)?;
        let mut bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                bias: true,
                dropout: 0.1,
                layer_norm_enabled: true,
                eps: 1e-6,
            },
            vb,
        )?;

        let output_tensor = bit_attention.forward(&input_tensor, true).unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }
}
