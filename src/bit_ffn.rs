use candle_core::{Module, Tensor};
use candle_nn::{seq, Sequential, VarBuilder};

use crate::bit_linear::Bitlinear;

pub struct BitFeedForward {
    layer: Sequential,
}

impl BitFeedForward {
    #[allow(dead_code)]
    pub fn load(dim: usize, ff_mult: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_dim = dim * ff_mult;
        let layer = seq()
            .add(Bitlinear::load(dim, hidden_dim, vb.device())?)
            .add(Tensor::gelu)
            .add(Bitlinear::load(hidden_dim, dim, vb.device())?);
        Ok(Self { layer })
    }
}

impl Module for BitFeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use crate::utils_tensor::device;
    use candle_core::{DType, Device, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_applies_forward_pass_dim_2() -> Result<()> {
        let device: Device = device(true).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (10, dim), &device)?;
        let bff = super::BitFeedForward::load(dim, 4, vb)?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims2()?;

        assert_eq!(output_shape.0, 10);
        assert_eq!(output_shape.1, dim);

        Ok(())
    }

    #[test]
    fn it_applies_forward_pass_dim_3() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (1, 10, dim), &device)?;
        let bff = super::BitFeedForward::load(dim, 4, vb)?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims3()?;

        assert_eq!(output_shape.0, 1);
        assert_eq!(output_shape.1, 10);
        assert_eq!(output_shape.2, dim);

        Ok(())
    }
}
