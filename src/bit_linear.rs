use candle_core::Tensor;
use candle_nn::{
    init::{FanInOut, NonLinearity, NormalOrUniform},
    Init, Linear, Module, VarBuilder,
};

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
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let init_weights = Init::Kaiming {
            dist: NormalOrUniform::Normal,
            fan: FanInOut::FanIn,
            non_linearity: NonLinearity::ReLU,
        };
        let weight = vb.get_with_hints((out_features, in_features), "weight", init_weights)?;
        Ok(Self {
            in_features,
            out_features,
            weight,
        })
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = sign(x)?;
        let gamma = self.weight.abs()?.mean_all()?; // 0 dimensional tensor
        let w_scaled = self.weight.broadcast_mul(&gamma)?; // 2 dimensional vector
        let w_quantized = sign(&w_scaled)?.mul(&w_scaled.abs()?.round()?.clamp(0u8, 1u8)?)?;
        let x: Tensor = Linear::new(w_quantized, None).forward(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use crate::utils_tensor::device;
    use candle_core::{DType, Module, Result, Tensor};
    use candle_nn::var_builder::VarBuilderArgs;
    use test::Bencher;

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilderArgs::zeros(DType::F32, &device.clone());
        let in_features = 128;
        let out_features = 64;
        let bl = super::Bitlinear::load(in_features, out_features, vb)?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (10, 128), &device.clone())?;
        let output = bl.forward(&input).unwrap();
        let output_shape = output.shape().dims2()?;

        assert_eq!(output_shape.0, 10);
        assert_eq!(output_shape.1, 64);

        Ok(())
    }

    #[bench]
    fn bench_bit_linear(b: &mut Bencher) -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilderArgs::zeros(DType::F32, &device.clone());
        let in_features = 128;
        let out_features = 64;
        let bl = super::Bitlinear::load(in_features, out_features, vb)?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (10, 128), &device)?;

        b.iter(|| {
            for _ in 1..100 {
                bl.forward(&input).unwrap();
            }
        });

        Ok(())
    }
}
