use crate::rms_norm::RMSNorm;
use crate::transformer::Transformer;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, seq, var_builder, Embedding, Sequential};
use candle_transformers::quantized_nn::linear;
use candle_transformers::quantized_var_builder;

struct BitNetTransformer {
    embedding: Embedding,
    transformer: Transformer,
    to_logits: Sequential,
}

impl BitNetTransformer {
    pub fn new(
        dim: usize,
        heads: usize,
        depth: usize,
        ff_mult: usize,
        num_tokens: usize,
        device: &Device,
        vb: var_builder::VarBuilder,
        qvb: quantized_var_builder::VarBuilder,
    ) -> Result<Self> {
        let transformer = Transformer::new(dim, heads, depth, ff_mult, vb.clone())?;
        let to_logits = seq()
            .add(RMSNorm::new(dim, device)?)
            .add(linear(dim, num_tokens, qvb)?);
        let embedding = embedding(num_tokens, dim, vb.clone())?;
        Ok(Self {
            transformer,
            to_logits,
            embedding,
        })
    }
}

impl Module for BitNetTransformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let output = self.embedding.forward(x)?;
        let output = self.transformer.forward(&output)?;
        let output = self.to_logits.forward(&output)?;
        Ok(output)
    }
}
