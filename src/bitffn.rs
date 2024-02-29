use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{seq, Sequential};

use crate::bitlinear::Bitlinear;

pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

pub struct BitFeedForward {
    #[allow(dead_code)]
    dim: usize,
    #[allow(dead_code)]
    ff_mult: usize,
    layer: Sequential,
    pub first_out_features: usize,
    pub third_in_features: usize,
}

impl BitFeedForward {
    #[allow(dead_code)]
    pub fn load(dim: usize, ff_mult: usize, device: &Device) -> Result<Self> {
        let hidden_dim = dim * ff_mult;
        let first = Bitlinear::load(dim, hidden_dim, &device)?;
        let second = relu;
        let third = Bitlinear::load(hidden_dim, dim, &device)?;
        let first_out_features = first.out_features;
        let third_in_features = third.in_features;
        let layer = seq().add(first).add(second).add(third);
        Ok(Self {
            dim: dim,
            ff_mult: ff_mult,
            layer: layer,
            first_out_features: first_out_features,
            third_in_features: third_in_features,
        })
    }
}

impl Module for BitFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer.forward(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use candle_core::{DType, Device, Module, Result, Tensor};

    #[test]
    fn it_loads() -> Result<()> {
        let bl = super::BitFeedForward::load(512, 4, &Device::Cpu)?;
        assert_eq!(bl.dim, 512);
        assert_eq!(bl.ff_mult, 4);
        assert_eq!(bl.layer.len(), 3);
        assert_eq!(bl.first_out_features, 512 * 4);
        assert_eq!(bl.third_in_features, 512 * 4);
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let bff = super::BitFeedForward::load(512, 4, &Device::Cpu)?;
        let input = Tensor::ones((1, 512), DType::F32, &Device::Cpu)?;
        let output = bff.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 512]);
        Ok(())
    }
}
