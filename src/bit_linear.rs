use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::utils::sign;

pub struct Bitlinear {
    pub in_features: usize,
    pub out_features: usize,
    weight: Tensor,
}

impl Bitlinear {
    pub fn load(in_features: usize, out_features: usize, device: &Device) -> Result<Self> {
        let weight: Tensor = Tensor::randn(0f32, 1f32, (out_features, in_features), device)?;
        Ok(Self {
            in_features: in_features,
            out_features: out_features,
            weight: weight,
        })
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("weight: {:?}", self.weight);
        let x = sign(x)?;
        println!("x: {:?}", x);
        let gamma = self.weight.abs()?.mean_all()?; // 0 dimensional tensor
        println!("gamma: {:?}", gamma);
        let w_scaled = self.weight.broadcast_mul(&gamma)?; // 2 dimensional vector
        println!("w_scaled: {:?}", w_scaled);
        let w_quantized = sign(&w_scaled)?.mul(&w_scaled.abs()?.round()?.clamp(0u8, 1u8)?)?;
        println!("w_quantized: {:?}", w_quantized);
        let output = Linear::new(w_quantized, None).forward(&x)?;
        println!("output: {:?}", output);
        Ok(output)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use candle_core::{Module, Result, Tensor};

    use crate::utils::device;

    #[test]
    fn it_loads_with_provided_options() -> Result<()> {
        let device = device(false)?;
        let bl = super::Bitlinear::load(3, 3, &device)?;
        assert!(bl.in_features == 3);
        assert!(bl.out_features == 3);
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let device = device(true)?;
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
