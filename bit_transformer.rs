use std::usize;

use crate::bit_attention::BitAttention;
use crate::bit_ffn::BitFeedForward;
use crate::utils_rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{embedding, linear, seq, Embedding, Sequential, VarBuilder};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub struct Transformer {
    attn_layers: Vec<BitAttention>,
    ffn_layers: Vec<BitFeedForward>,
}

impl Transformer {
    pub fn new(
        _num_tokens: usize,
        dim: usize,
        heads: usize,
        depth: usize,
        ff_mult: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut attn_layers = Vec::new();
        let mut ffn_layers = Vec::new();

        for _ in 0..(depth) {
            attn_layers.push(BitAttention::load(
                dim,
                heads,
                4,
                0.1,
                true,
                true,
                1e-6,
                1.0,
                vb.clone(),
            )?);
            ffn_layers.push(BitFeedForward::load(dim, ff_mult, vb.device())?);
        }

        Ok(Self {
            attn_layers,
            ffn_layers,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (attn, ffn) in self.attn_layers.iter_mut().zip(self.ffn_layers.iter()) {
            (x, _) = attn.forward(x.clone(), x.clone(), x.clone(), false, true, false)?;
            x = x.add(&x)?;
            x = (ffn.forward(&x) + x)?
        }
        Ok(x)
    }
}

struct BitTransformer {
    embedding: Embedding,
    transformer: Transformer,
    to_logits: Sequential,
}

impl BitTransformer {
    pub fn load(
        dim: usize,
        depth: usize,
        num_tokens: usize,
        heads: usize,
        ff_mult: usize,
        device: &Device,
    ) -> Result<Self> {
        let t_vb = VarBuilder::zeros(candle_core::DType::F32, &device.clone());
        let e_vb = VarBuilder::zeros(candle_core::DType::F32, &device.clone());
        let embedding = embedding(num_tokens, dim, e_vb.pp("weight"))?;
        println!("embedding: {:?}", embedding);
        let transformer = Transformer::new(num_tokens, dim, heads, depth, ff_mult, t_vb.clone())?;
        let to_logits = seq()
            .add(RmsNorm::load(1e-6, dim, t_vb.clone())?)
            .add(linear(dim, num_tokens, e_vb.clone())?);
        Ok(Self {
            transformer,
            to_logits,
            embedding,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        println!("x: {:?}", x);
        let x = self.embedding.forward(x)?;
        println!("x: {:?}", x);
        let x = self.transformer.forward(&x)?;
        println!("x: {:?}", x);
        let x = self.to_logits.forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bitnet_transformer_tests {
    use crate::utils_tensor::device;

    use super::BitTransformer;
    use anyhow::Result;
    use candle_core::Tensor;

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let dtype = candle_core::DType::U32;
        let device = &device(false)?;
        let mut t = BitTransformer::load(1024, 6, 20000, 8, 4, device)?;
        let x = Tensor::ones((1, 1024), dtype, device)?;
        let x = t.forward(&x)?;

        assert_eq!(x.shape().dims(), &[1, 1024, 20000]);

        Ok(())
    }
}
