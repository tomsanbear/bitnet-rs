use std::usize;

use crate::bit_attention::BitAttention;
use crate::bit_ffn::BitFeedForward;
use crate::utils_rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use candle_nn::{embedding, linear, seq, var_builder, Embedding, Sequential};

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
        println!("x: {:?}", x.shape().dims());
        for (attn, ffn) in self.attn_layers.iter_mut().zip(self.ffn_layers.iter()) {
            (x, _) = attn.forward(x.clone(), x.clone(), x.clone(), false, true, false)?;
            x = x.add(&x)?;
            x = (ffn.forward(&x) + x)?
        }
        Ok(x)
    }
}

struct BitTransformer {
    embed_tokens: Embedding,
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
        vb: var_builder::VarBuilder,
    ) -> Result<Self> {
        let embed_tokens = embedding(num_tokens, dim, vb.pp("transformer.word_embeddings"))?;
        let transformer = Transformer::new(num_tokens, dim, heads, depth, ff_mult, vb.clone())?;
        let to_logits = seq().add(RmsNorm::load(1e-6, dim, vb.clone())?).add(linear(
            dim,
            num_tokens,
            vb.clone(),
        )?);
        Ok(Self {
            transformer,
            to_logits,
            embed_tokens,
        })
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let output = self.embed_tokens.forward(x)?;
        let output = self.transformer.forward(&output)?;
        let output = self.to_logits.forward(&output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod bitnet_transformer_tests {
    use crate::utils_tensor::device;

    use super::BitTransformer;
    use anyhow::Result;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu);
        BitTransformer::load(512, 6, 20000, 8, 4, vb)?;
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let dtype = candle_core::DType::BF16;
        let device = &device(false)?;
        let vb = VarBuilder::zeros(dtype, device);
        let mut t = BitTransformer::load(512, 6, 20000, 8, 4, vb)?;
        let x = Tensor::ones(256, dtype, device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        t.forward(&x)?;
        Ok(())
    }
}
