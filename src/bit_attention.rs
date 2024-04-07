use crate::{
    bit_linear::{Bitlinear, BitlinearCfg},
    utils_tensor::scaled_dot_product_attention,
};
use anyhow::{anyhow, Result};
use candle_core::Tensor;
use candle_einops::einops;
use candle_nn::VarBuilder;
use tracing::instrument;

#[derive(Debug, Clone, Copy)]
pub struct BitAttentionCfg {
    pub dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub dropout: f32,
    pub eps: f64,
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

        Ok(BitAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            dropout: cfg.dropout,
        })
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

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
        let bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                dropout: 0.1,
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
        let bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                dropout: 0.1,
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
        let bit_attention = BitAttention::load(
            BitAttentionCfg {
                dim: 64,
                n_heads: 8,
                n_kv_heads: 8,
                dropout: 0.1,
                eps: 1e-6,
            },
            vb,
        )?;

        let output_tensor = bit_attention.forward(&input_tensor, true).unwrap();

        assert_eq!(output_tensor.shape().dims(), &[2, 8, 64]);

        Ok(())
    }
}
