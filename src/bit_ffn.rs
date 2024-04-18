use crate::bit_dropout::{Dropout, DropoutCfg};
use crate::bit_linear::{Bitlinear, BitlinearCfg};
use crate::rms_norm::RmsNorm;
use candle_core::{Module, Tensor};
use candle_nn::{Activation, VarBuilder};
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
    proj_in: Bitlinear,
    activation: Activation,
    post_act_norm: RmsNorm,
    dropout: Dropout,
    proj_out: Bitlinear,
}

impl BitFeedForward {
    pub fn load(cfg: BitFeedForwardCfg, vb: VarBuilder) -> anyhow::Result<Self> {
        // Setup internal parameters
        let inner_dim = cfg.dim * cfg.ff_mult;

        // Use swiglu from 1.58 paper
        let activation = Activation::Swiglu;

        // Dropout layer, if train is passed then this is skipped
        let dropout = Dropout::load(DropoutCfg {
            p: cfg.dropout,
            is_training: cfg.train,
        })?;

        // Post activation normalization
        let post_act_norm = RmsNorm::load(cfg.eps, inner_dim, vb.pp("norm"))?;

        // Input linear layer
        let proj_in = Bitlinear::load(
            BitlinearCfg {
                in_features: cfg.dim,
                out_features: inner_dim * 2,
                eps: cfg.eps,
            },
            vb.pp("proj"),
        )?;

        // Linear layer
        let proj_out = Bitlinear::load(
            BitlinearCfg {
                in_features: inner_dim,
                out_features: cfg.dim,
                eps: cfg.eps,
            },
            vb.pp("linear"),
        )?;

        // Return the layer as a sequential module
        Ok(Self {
            proj_in,
            activation,
            post_act_norm,
            dropout,
            proj_out,
        })
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.proj_in.forward(x)?;
        let x = self.activation.forward(&x)?;
        let x = self.post_act_norm.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        let x = self.proj_out.forward(&x)?;
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
