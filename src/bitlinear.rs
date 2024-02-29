use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, Module};

use crate::utils::sign;

pub struct Bitlinear {
    pub in_features: usize,
    pub out_features: usize,
    weight: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl Bitlinear {
    pub fn load(in_features: usize, out_features: usize, device: &Device) -> Result<Self> {
        let weight: Tensor = Tensor::randn(0f32, 1f32, (out_features, in_features), device)?;
        let gamma = Tensor::ones(in_features, weight.dtype(), device)?;
        let beta = Tensor::ones(out_features, weight.dtype(), device)?;
        Ok(Self {
            in_features: in_features,
            out_features: out_features,
            weight: weight,
            gamma: gamma,
            beta: beta,
            eps: 1e-5,
        })
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply layer normalization
        // Torch uses ones as a default if the weight arg is omitted
        let input_norm = LayerNorm::new_no_bias(x.ones_like()?, self.eps).forward(x)?;

        // Absmax quantization
        // The input_norm is currently a (1,1) we need to expand it to (1, in_features) to match the operations that follow
        let quant_scale =
            input_norm
                .abs()?
                .max_keepdim(1)?
                .pad_with_same(1, 0, self.in_features - 1)?;
        let gamma = self.gamma.clone();
        let input_quant = (quant_scale / gamma.unsqueeze(0))?;
        let input_quant = (sign(&input_norm)? * input_quant)?;
        println!("{:?}", input_quant.shape());

        // 1 bit weight quantization
        let weight_quant = sign(&self.weight)?;
        println!("{:?}", weight_quant.shape());

        // MatMul with 1-bit weights using matmul for explicit operation
        let output = input_quant.matmul(&weight_quant.t()?)?;
        println!("{:?}", output.shape());

        // Dequantize with learnable parameters
        let beta = self.beta.unsqueeze(0)?.broadcast_as(output.shape())?;
        let output = output.mul(&beta)?;

        Ok(output)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use candle_core::{Device, Module, Result, Tensor};

    #[test]
    fn it_loads_with_provided_options() -> Result<()> {
        let bl = super::Bitlinear::load(3, 3, &Device::Cpu)?;
        assert!(bl.in_features == 3);
        assert!(bl.out_features == 3);
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let bl = super::Bitlinear::load(100, 256, &Device::Cpu)?;

        let tensor = Tensor::randn(0f32, 1f32, (1, 100), &Device::Cpu)?;

        let output = bl.forward(&tensor).unwrap();
        assert_eq!(output.shape().dims(), &[1, 256]);
        Ok(())
    }
}
