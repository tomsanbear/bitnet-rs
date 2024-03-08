use crate::bit_attention::{BitAttention, BitAttentionCfg};
use crate::bit_ffn::{BitFeedForward, BitFeedForwardCfg};
use crate::config::Config;
use crate::rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{embedding, linear, seq, Embedding, Sequential, VarBuilder};

pub struct BitTransformer {
    embedding: Embedding,
    blocks: Vec<(BitAttention, BitFeedForward)>,
    to_logits: Sequential,
    span: tracing::Span,
}

impl BitTransformer {
    pub fn load(cfg: Config, vb: VarBuilder, train: bool) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "bit-transformer");
        let embedding = embedding(cfg.vocab_size, cfg.dim, vb.pp("embedding"))?;
        let blocks: Vec<_> = (0..(cfg.depth))
            .map(|i| {
                (
                    BitAttention::load(
                        BitAttentionCfg {
                            embed_dim: cfg.dim,
                            query_heads: cfg.heads,
                            kv_heads: 4,
                            dropout: 0.1,
                            layer_norm_enabled: true,
                            eps: cfg.eps,
                        },
                        vb.pp(&format!("attn.{i}")),
                    )
                    .unwrap(),
                    BitFeedForward::load(
                        BitFeedForwardCfg {
                            dim: cfg.dim,
                            ff_mult: cfg.ff_mult,
                            dropout: cfg.ff_dropout,
                            train,
                            eps: cfg.eps,
                        },
                        vb.pp(&format!("ffn.{i}")),
                    )
                    .unwrap(),
                )
            })
            .collect();

        let to_logits = seq()
            .add(RmsNorm::load(cfg.eps, cfg.dim, vb.pp("rms_norm"))?)
            .add(linear(cfg.dim, cfg.vocab_size, vb.pp("logits_linear"))?);

        Ok(Self {
            span,
            blocks,
            to_logits,
            embedding,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Run the embedding layer
        let x_embed = self.embedding.forward(x)?;

        // Fold each block forward
        let x = self.blocks.iter().fold(x_embed.clone(), |x, (attn, ffn)| {
            let (x, _) = attn
                .forward(x.clone(), x.clone(), x.clone(), false, true, false)
                .unwrap();
            let x = x.add(&x_embed).unwrap();
            let x = ffn.forward(&x).unwrap();
            x.add(&x).unwrap()
        });

        // Convert to logits
        let x = self.to_logits.forward(&x)?;

        // Return the logits
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

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let device = &device(false)?;
        let vb = VarBuilder::zeros(DType::F32, device);
        let mut t = BitTransformer::load(Config::default(), vb, true)?;
        let x = Tensor::ones((1, 128), DType::U32, device)?;
        let x = t.forward(&x)?;

        assert_eq!(x.shape().dims(), &[1, 128, 32000]);

        Ok(())
    }
}
