use crate::bit_dropout::{Dropout, DropoutCfg};
use crate::bit_linear::Bitlinear;
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, seq, Activation, LayerNormConfig, Sequential, VarBuilder};

pub struct BitFeedForwardCfg {
    pub dim: usize,
    pub ff_mult: usize,
    pub dropout: f32,
    pub train: bool,
    pub eps: f32,
}

pub struct BitFeedForward {
    layer: Sequential,
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

        // Setup the GLU function
        let proj = seq()
            .add(Bitlinear::load(
                cfg.dim,
                inner_dim,
                1,
                8,
                cfg.eps,
                vb.pp("proj"),
            )?)
            .add(activation);

        // Linear layer
        let linear = Bitlinear::load(inner_dim, cfg.dim, 1, 8, cfg.eps, vb.pp("linear"))?;

        // Return the layer as a sequential module
        Ok(Self {
            layer: seq().add(proj).add(norm).add(dropout).add(linear),
        })
    }
}

impl Module for BitFeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use super::BitFeedForward;
    use crate::{bit_ffn::BitFeedForwardCfg, utils_tensor::device};
    use candle_core::{DType, Device, Module, Tensor};
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
