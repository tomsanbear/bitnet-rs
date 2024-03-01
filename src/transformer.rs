use std::sync::Arc;

use crate::bitffn::BitFeedForward;
use crate::config::Config;
use crate::{attention::Attention, rotary_embedding::RotaryEmbedding};
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct Transformer {
    attn_layers: Vec<Attention>,
    ffn_layers: Vec<BitFeedForward>,
}

impl Transformer {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let mut attn_layers = Vec::new();
        let mut ffn_layers = Vec::new();
        let re = Arc::new(RotaryEmbedding::new(vb.dtype(), &config, vb.device())?);

        // Layers are multihead attention blocks
        for _ in 0..(config.num_hidden_layers) {
            attn_layers.push(Attention::new(re.clone(), &config, vb.clone())?);
            ffn_layers.push(BitFeedForward::load(config.hidden_size, 1, vb.device())?);
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
            x = (attn.forward(&x.clone())? + x)?;
            println!("x: {:?}", x.shape().dims());
            x = (ffn.forward(&x)? + x)?;
            println!("x: {:?}", x.shape().dims());
        }
        Ok(x)
    }
}

#[cfg(test)]
mod transformer_tests {
    use candle_core::{Device, Result};
    use candle_nn::VarBuilder;

    use super::Transformer;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let config = crate::config::Config {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: 256,
            use_flash_attn: false,
        };
        let t = Transformer::new(config, vb)?;

        assert_eq!(t.attn_layers.len(), 4);
        assert_eq!(t.ffn_layers.len(), 4);

        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let config = crate::config::Config {
            vocab_size: 32000,
            hidden_size: 256,
            intermediate_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.,
            sliding_window: 256,
            use_flash_attn: false,
        };
        let mut t = Transformer::new(config, vb)?;

        let input = candle_core::Tensor::randn(0f64, 1f64, (256,), &Device::Cpu)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let output = t.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 1, 256]);

        Ok(())
    }
}
