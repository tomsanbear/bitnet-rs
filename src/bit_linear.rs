use anyhow::Ok;
use candle_core::{Tensor, D};
use candle_nn::{Init, Module, VarBuilder};
use candle_transformers::models::with_tracing::{Linear, RmsNorm};
use tracing::instrument;

#[derive(Debug, Clone, Copy)]
pub struct BitlinearCfg {
    pub in_features: usize,
    pub out_features: usize,
    pub eps: f64,
}

#[derive(Debug)]
pub struct Bitlinear {
    weight: Tensor,
    layer_norm: RmsNorm,
}

impl Bitlinear {
    pub fn load(cfg: BitlinearCfg, vb: VarBuilder) -> anyhow::Result<Self> {
        let weight = vb.get_with_hints(
            (cfg.out_features, cfg.in_features),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let layer_norm = RmsNorm::new(cfg.in_features, cfg.eps, vb.pp("rms_norm"))?;
        Ok(Self { weight, layer_norm })
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        fn activation_quant(x: &Tensor) -> anyhow::Result<Tensor> {
            let scale = (127.0
                / x.abs()?
                    .max(D::Minus1)?
                    .max(D::Minus1)?
                    .clamp(1e-5, f32::INFINITY)?)?;
            let y = x
                .broadcast_mul(&scale.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?)?
                .clamp(-128.0, 127.0)?;
            Ok(y)
        }

        fn weight_quant(x: &Tensor) -> anyhow::Result<Tensor> {
            let scale = x.abs()?.mean_all()?;
            let e = x.mean_all()?;
            let u = x.broadcast_sub(&e)?.sign()?.broadcast_mul(&scale)?;
            Ok(u)
        }

        let weight = self.weight.clone();

        let x_norm = self.layer_norm.forward(x)?;

        let x_quant = (x_norm.clone() + (activation_quant(&x_norm)? - x_norm)?.detach())?;

        let w_quant = (weight.clone() + (weight_quant(&weight)? - weight)?.detach())?;

        let y = Linear::from_weights(w_quant, None).forward(&x_quant)?;

        Ok(y)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use super::Bitlinear;
    use crate::{bit_linear::BitlinearCfg, utils_tensor::device};
    use candle_core::{DType, Tensor};
    use candle_nn::var_builder::VarBuilderArgs;

    #[test]
    fn it_applies_forward_pass() -> anyhow::Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilderArgs::zeros(DType::F32, &device.clone());
        let in_features = 64;
        let out_features = 64;
        let bl = Bitlinear::load(
            BitlinearCfg {
                in_features,
                out_features,
                eps: 1e-6,
            },
            vb,
        )?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (1, 64), &device.clone())?;
        let output = bl.forward(&input)?;
        assert_eq!(output.shape().dims2()?, (1, 64));
        Ok(())
    }
}
