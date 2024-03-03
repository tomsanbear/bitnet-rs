use std::usize;

use crate::bit_attention::BitAttention;
use crate::bit_ffn::BitFeedForward;
use crate::utils_rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{embedding, linear, seq, Embedding, Sequential, VarBuilder};

pub struct Transformer {
    attn_layers: Vec<BitAttention>,
    ffn_layers: Vec<BitFeedForward>,
}

impl Transformer {
    pub fn new(
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
                1e-6,
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
            x = (ffn.forward(&x) + &x)?
        }
        Ok(x)
    }
}

pub struct BitTransformer {
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
        let transformer = Transformer::new(dim, heads, depth, ff_mult, t_vb.clone())?;
        let to_logits = seq()
            .add(RmsNorm::load(1e-6, dim, t_vb.clone())?)
            .add(linear(dim, num_tokens, e_vb.clone())?);
        Ok(Self {
            transformer,
            to_logits,
            embedding,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let x = self.embedding.forward(&x)?;
        let x = self.transformer.forward(&x)?;
        let x = self.to_logits.forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bitnet_transformer_tests {
    use crate::utils_tensor::device;

    use super::BitTransformer;
    use anyhow::Result;
    use candle_core::{DType, Tensor};
    use test::Bencher;

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let dtype = candle_core::DType::U32;
        let device = &device(false)?;
        let mut t = BitTransformer::load(128, 6, 20000, 8, 4, device)?;
        let x = Tensor::ones((1, 128), dtype, device)?;
        let x = t.forward(&x)?;

        assert_eq!(x.shape().dims(), &[1, 128, 20000]);

        Ok(())
    }

    #[bench]
    fn bench_bit_transformer(b: &mut Bencher) -> Result<()> {
        let device = &device(false)?;

        b.iter(|| {
            for _ in 1..10 {
                let mut t = BitTransformer::load(128, 6, 20000, 8, 4, device).unwrap();
                let x = Tensor::ones((1, 128), DType::U32, device).unwrap();
                t.forward(&x).unwrap();
            }
        });

        Ok(())
    }
}
