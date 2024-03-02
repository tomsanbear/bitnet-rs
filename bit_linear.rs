use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

use crate::utils_tensor::sign;

pub struct Bitlinear {
    pub in_features: usize,
    pub out_features: usize,
    weight: Tensor,
}

impl Bitlinear {
    pub fn load(
        in_features: usize,
        out_features: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let weight: Tensor = Tensor::randn(0f32, 1f32, (out_features, in_features), device)?;
        Ok(Self {
            in_features: in_features,
            out_features: out_features,
            weight: weight,
        })
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = sign(x)?;
        let gamma = self.weight.abs()?.mean_all()?; // 0 dimensional tensor
        let w_scaled = self.weight.broadcast_mul(&gamma)?; // 2 dimensional vector
        let w_quantized = sign(&w_scaled)?.mul(&w_scaled.abs()?.round()?.clamp(0u8, 1u8)?)?;
        Linear::new(w_quantized, None).forward(&x)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use candle_core::{Module, Result, Tensor};

    use crate::utils_tensor::device;

    #[test]
    fn it_loads_with_provided_options() -> Result<()> {
        let device = device(false).unwrap();
        let bl = super::Bitlinear::load(3, 3, &device)?;
        assert!(bl.in_features == 3);
        assert!(bl.out_features == 3);
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let device = device(true).unwrap();
        let in_features = 128;
        let out_features = 64;
        let bl = super::Bitlinear::load(in_features, out_features, &device)?;
        let input: Tensor = Tensor::randn(0f32, 1.0, (10, 128), &device)?;
        let output = bl.forward(&input).unwrap();
        println!("output: {}", output);

        let output_shape = output.shape().dims2()?;

        assert_eq!(output_shape.0, 10);
        assert_eq!(output_shape.1, 64);

        Ok(())
    }
}
