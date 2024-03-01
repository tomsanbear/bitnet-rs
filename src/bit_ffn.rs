use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{seq, Sequential};

use crate::bit_linear::Bitlinear;

pub struct BitFeedForward {
    layer: Sequential,
}

impl BitFeedForward {
    #[allow(dead_code)]
    pub fn load(dim: usize, ff_mult: usize, device: &Device) -> Result<Self> {
        let hidden_dim = dim * ff_mult;
        let layer = seq()
            .add(Bitlinear::load(dim, hidden_dim, &device)?)
            .add(Tensor::gelu)
            .add(Bitlinear::load(hidden_dim, dim, &device)?);
        Ok(Self { layer: layer })
    }
}

impl Module for BitFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer.forward(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use candle_core::{Device, Module, Result, Tensor};

    use crate::utils::device;

    #[test]
    fn it_loads() -> Result<()> {
        let bl = super::BitFeedForward::load(512, 4, &Device::Cpu)?;
        assert_eq!(bl.layer.len(), 3);
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass_dim_2() -> Result<()> {
        let device = device(true)?;
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (10, dim), &device)?;
        let bff = super::BitFeedForward::load(dim, 4, &Device::Cpu)?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims2()?;

        assert_eq!(output_shape.0, 10);
        assert_eq!(output_shape.1, dim);

        Ok(())
    }

    #[test]
    fn it_applies_forward_pass_dim_3() -> Result<()> {
        let device = device(true)?;
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (1, 10, dim), &device)?;
        let bff = super::BitFeedForward::load(dim, 4, &Device::Cpu)?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims3()?;

        assert_eq!(output_shape.0, 1);
        assert_eq!(output_shape.1, 10);
        assert_eq!(output_shape.2, dim);

        Ok(())
    }
}
