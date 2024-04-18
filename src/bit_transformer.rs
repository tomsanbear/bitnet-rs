use crate::bit_attention::{BitAttention, BitAttentionCfg};
use crate::bit_ffn::{BitFeedForward, BitFeedForwardCfg};
use crate::config::Config;
use crate::embedding::Embedding;
use crate::rms_norm::RmsNorm;
use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{linear, Linear};
use tracing::instrument;

#[derive(Debug)]
pub struct BitTransformer {
    embedding: Embedding,
    blocks: Vec<(BitAttention, BitFeedForward)>,
    rms_norm: RmsNorm,
    logits_linear: Linear,
}

impl BitTransformer {
    pub fn load(cfg: Config, vb: VarBuilder, train: bool) -> Result<Self> {
        let embedding = Embedding::new(cfg.vocab_size, cfg.dim, vb.pp("embedding"))?;
        let blocks: Vec<_> = (0..(cfg.depth))
            .map(|i| {
                (
                    BitAttention::load(
                        BitAttentionCfg {
                            dim: cfg.dim,
                            n_heads: cfg.heads,
                            n_kv_heads: 8,
                            dropout: 0.1,
                            eps: cfg.eps,
                            max_seq_len: cfg.seq_len,
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

        let rms_norm = RmsNorm::load(cfg.eps, cfg.dim, vb.pp("rms_norm"))?;
        let logits_linear = linear(cfg.dim, cfg.vocab_size, vb.pp("logits_linear"))?;

        Ok(Self {
            blocks,
            rms_norm,
            logits_linear,
            embedding,
        })
    }

    #[instrument]
    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // Run the embedding layer
        let x_embed = self.embedding.forward(x)?;

        // Fold each block forward
        let mut x = x_embed.clone();
        for (attn, ffn) in self.blocks.iter_mut() {
            x = attn.forward(&x, true, index_pos)?;
            x = x.add(&x_embed)?;
            x = ffn.forward(&x)?;
            x = x.add(&x)?;
        }

        // Convert to logits
        let x = self.rms_norm.forward(&x)?;
        let x = self.logits_linear.forward(&x)?;
        Ok(x)
    }
}
