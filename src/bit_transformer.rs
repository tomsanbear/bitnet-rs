use crate::config::Config;
use crate::rms_norm::RmsNorm;
use crate::transformer::Transformer;
use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, linear, seq, var_builder, Embedding, Sequential};

struct BitTransformer {
    embed_tokens: Embedding,
    transformer: Transformer,
    to_logits: Sequential,
}

impl BitTransformer {
    pub fn load(config: Config, vb: var_builder::VarBuilder) -> Result<Self> {
        let transformer = Transformer::new(config, vb.clone())?;
        let to_logits = seq()
            .add(RmsNorm::load(config.vocab_size, vb.clone())?)
            .add(linear(config.vocab_size, config.hidden_size, vb.clone())?);
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
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
    use crate::utils::device;

    use super::BitTransformer;
    use candle_core::{Device, Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F32, &Device::Cpu);
        let config = crate::config::Config {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 0.03,
            sliding_window: 256,
            use_flash_attn: true,
        };
        BitTransformer::load(config, vb)?;
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let dtype = candle_core::DType::U32;
        let device = &device(false)?;
        let vb = VarBuilder::zeros(dtype, device);
        let config = crate::config::Config {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 0.03,
            sliding_window: 256,
            use_flash_attn: true,
        };
        let mut t = BitTransformer::load(config, vb)?;
        let x = Tensor::ones(256, dtype, device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        t.forward(&x)?;
        Ok(())
    }
}
