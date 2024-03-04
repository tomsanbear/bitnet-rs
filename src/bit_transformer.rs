use crate::bit_attention::BitAttention;
use crate::bit_ffn::BitFeedForward;
use crate::config::Config;
use crate::utils_rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{embedding, linear, seq, Embedding, Sequential, VarBuilder};

pub struct BitTransformer {
    embedding: Embedding,
    blocks: Vec<(BitAttention, BitFeedForward)>,
    to_logits: Sequential,
}

impl BitTransformer {
    pub fn load(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(cfg.vocab_size, cfg.dim, vb.pp("model.embed_tokens"))?;

        let blocks: Vec<_> = (0..(cfg.depth))
            .map(|i| {
                (
                    BitAttention::load(
                        cfg.dim,
                        cfg.heads,
                        4,
                        0.1,
                        true,
                        1e-6,
                        vb.pp(&format!("model.attn_layers.{i}")),
                    )
                    .unwrap(),
                    BitFeedForward::load(
                        cfg.dim,
                        cfg.ff_mult,
                        vb.pp(&format!("model.ffn_layers.{i}")),
                    )
                    .unwrap(),
                )
            })
            .collect();

        let to_logits = seq()
            .add(RmsNorm::load(1e-6, cfg.dim, vb.pp("model.norm"))?)
            .add(linear(cfg.dim, cfg.vocab_size, vb.pp("lm_head"))?);

        Ok(Self {
            blocks,
            to_logits,
            embedding,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.embedding.forward(x)?;
        for (attn, ffn) in self.blocks.iter() {
            (x, _) = attn.forward(x.clone(), x.clone(), x.clone(), false, true, false)?;
            x = x.add(&x)?;
            x = (ffn.forward(&x) + &x)?
        }
        let x = self.to_logits.forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bitnet_transformer_tests {
    use crate::{config::Config, utils_tensor::device};

    use super::BitTransformer;
    use anyhow::Result;
    use candle_core::{DType, Tensor};
    use candle_nn::VarBuilder;
    use test::Bencher;

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let dtype: DType = candle_core::DType::U32;
        let device = &device(false)?;
        let vb = VarBuilder::zeros(dtype, device);
        let mut t = BitTransformer::load(Config::default(), vb)?;
        let x = Tensor::ones((1, 128), dtype, device)?;
        let x = t.forward(&x)?;

        assert_eq!(x.shape().dims(), &[1, 128, 20000]);

        Ok(())
    }

    #[bench]
    fn bench_bit_transformer(b: &mut Bencher) -> Result<()> {
        let device = &device(false)?;
        let dtype: DType = candle_core::DType::U32;

        b.iter(|| {
            for _ in 1..10 {
                let vb = VarBuilder::zeros(dtype, device);
                let mut t = BitTransformer::load(Config::default(), vb).unwrap();
                let x = Tensor::ones((1, 128), DType::U32, device).unwrap();
                t.forward(&x).unwrap();
            }
        });

        Ok(())
    }
}
