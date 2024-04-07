use crate::bit_dropout::{Dropout, DropoutCfg};
use crate::bit_linear::{Bitlinear, BitlinearCfg};
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, Activation, LayerNorm, LayerNormConfig, VarBuilder};
use tracing::instrument;

pub struct BitFeedForwardCfg {
    pub dim: usize,
    pub ff_mult: usize,
    pub dropout: f32,
    pub train: bool,
    pub eps: f64,
}

#[derive(Debug)]
pub struct BitFeedForward {
    glu_linear: Bitlinear,
    activation: Activation,
    norm: LayerNorm,
    dropout: Dropout,
    linear: Bitlinear,
}

impl BitFeedForward {
    pub fn load(cfg: BitFeedForwardCfg, vb: VarBuilder) -> anyhow::Result<Self> {
        // Setup internal parameters
        let inner_dim = cfg.dim * cfg.ff_mult;

        // GELU is used as activation function
        // The original implementation has the option for SiLU, look into adding that at some point
        let activation = Activation::Gelu;

        // Dropout layer, if train is passed then this is skipped
        let dropout = Dropout::load(DropoutCfg {
            p: cfg.dropout,
            is_training: cfg.train,
        })?;

        // Layer normalization
        let norm = layer_norm(
            inner_dim,
            LayerNormConfig {
                eps: cfg.eps.into(),
                ..LayerNormConfig::default()
            },
            vb.pp("norm"),
        )?;

        let glu_linear = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: inner_dim,
                eps: cfg.eps,
            },
            vb.pp("proj"),
        )?;

        // Linear layer
        let linear = Bitlinear::load(
            BitlinearCfg {
                in_features: inner_dim,
                out_features: cfg.dim,
                eps: cfg.eps,
            },
            vb.pp("linear"),
        )?;

        // Return the layer as a sequential module
        Ok(Self {
            glu_linear,
            activation,
            norm,
            dropout,
            linear,
        })
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.glu_linear.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.norm.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        let x = self.linear.forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use super::BitFeedForward;
    use crate::{bit_ffn::BitFeedForwardCfg, utils_tensor::device};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_applies_forward_pass_dim_2() -> anyhow::Result<()> {
        let device: Device = device(true).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (10, dim), &device)?;
        let bff = BitFeedForward::load(
            BitFeedForwardCfg {
                dim,
                ff_mult: 4,
                dropout: 0.1,
                train: true,
                eps: 1e-6,
            },
            vb,
        )?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims2()?;
        assert_eq!(output_shape, (10, dim));
        Ok(())
    }
}
