use crate::{bit_linear::Bitlinear, utils_tensor::scaled_dot_product_gqa};
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{layer_norm, LayerNormConfig, Module, VarBuilder};

pub struct BitAttention {
    q_proj: Bitlinear,
    k_proj: Bitlinear,
    v_proj: Bitlinear,
    norm: Option<candle_nn::LayerNorm>,
    out_proj: Bitlinear,
    dropout: f32,
    head_dim: usize,
    query_heads: usize,
    kv_heads: usize,
    bias_enabled: bool,
    gamma_init: f32,
    device: Device,
    dtype: DType,
}

impl BitAttention {
    pub fn load(
        embed_dim: usize,
        query_heads: usize,
        kv_heads: usize,
        dropout: f32,
        bias_enabled: bool,
        layer_norm_enabled: bool,
        layer_norm_eps: f64,
        gamma_init: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let kv_embed_dim = embed_dim / query_heads * kv_heads;
        let head_dim = embed_dim / query_heads;
        if query_heads % kv_heads != 0 {
            return Err(anyhow!("query_heads must be divisible by kv_heads"));
        }
        if (embed_dim % query_heads) != 0 {
            return Err(anyhow!("embed_dim must be divisible by query_heads"));
        }
        if (embed_dim % kv_heads) != 0 {
            return Err(anyhow!("embed_dim must be divisible by kv_heads"));
        }
        if head_dim % 8 != 0 {
            return Err(anyhow!("head_dim must be divisible by 8"));
        }
        if head_dim > 128 {
            return Err(anyhow!("head_dim must be less than or equal to 128"));
        }

        let q_proj = Bitlinear::load(embed_dim, embed_dim, vb.device())?;
        let k_proj = Bitlinear::load(embed_dim, kv_embed_dim, vb.device())?;
        let v_proj = Bitlinear::load(embed_dim, kv_embed_dim, vb.device())?;

        let norm = match layer_norm_enabled {
            true => {
                let config = LayerNormConfig {
                    eps: layer_norm_eps,
                    ..LayerNormConfig::default()
                };
                Some(layer_norm(embed_dim, config, vb.clone())?)
            }
            false => None,
        };

        let out_proj = Bitlinear::load(kv_embed_dim, embed_dim, vb.device())?;

        // TODO: Original project makes a call to reset parameters, investigate why

        Ok(BitAttention {
            q_proj: q_proj,
            k_proj: k_proj,
            v_proj: v_proj,
            norm: norm,
            query_heads,
            kv_heads,
            head_dim,
            out_proj: out_proj,
            dropout: dropout,
            bias_enabled: bias_enabled,
            gamma_init: gamma_init,
            device: vb.device().clone(),
            dtype: vb.dtype(),
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
        println!("q: {:?}", q.dims());
        println!("k: {:?}", k.dims());
        println!("v: {:?}", v.dims());

        let (x, attn_weights) = scaled_dot_product_gqa(
            q,
            k,
            v,
            is_causal,
            need_weights,
            average_attn_weights,
            false,
            self.dropout,
            &self.device,
            self.dtype,
        )?;
        println!("x: {:?}", x.shape().dims());

        // x = rearrange(x, "b n h d -> b n (h d)")
        let x = x.permute([0, 1, 3, 2])?;

        Ok((x, attn_weights))
    }
}

#[cfg(test)]
mod bit_attention_tests {
    use crate::{bit_attention::BitAttention, utils_tensor::device};
    use candle_core::{Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_works() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(candle_core::DType::F32, &device);
        let bit_attention =
            BitAttention::load(512, 8, 4, 0.0, false, false, 1e-5, 1.0, vb).unwrap();
        let input = Tensor::randn(0f32, 1.0, (1, 10, 512), &device)?;
        let _out = bit_attention
            .forward(
                input.clone(),
                input.clone(),
                input.clone(),
                true,
                false,
                false,
            )
            .unwrap();
        Ok(())
    }
}
