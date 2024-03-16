use crate::utils_tensor::sign;
use anyhow::Ok;
use candle_core::{Tensor, D};
use candle_nn::{layer_norm, Init, LayerNorm, LayerNormConfig, Module, VarBuilder};
use candle_transformers::models::with_tracing::Linear;
use tracing::{event, span};

pub struct Bitlinear {
    num_groups: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    layer_norm: LayerNorm,
    eps: f32,
    q_b: f64,
    beta: Tensor,
    gamma: Tensor,
}

pub struct BitlinearCfg {
    pub in_features: usize,
    pub out_features: usize,
    pub num_groups: usize,
    pub b: i32,
    pub eps: f32,
    pub bias: bool,
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
        let layer_norm = layer_norm(
            cfg.in_features,
            LayerNormConfig {
                eps: cfg.eps.into(),
                ..LayerNormConfig::default()
            },
            vb.pp("layer_norm"),
        )?;
        let bias = match cfg.bias {
            true => Some(vb.get_with_hints(cfg.out_features, "bias", Init::Const(0.0))?),
            false => None,
        };
        let q_b = 2f64.powi(cfg.b - 1) as f64;
        let beta = Tensor::zeros(weight.dims()[0], weight.dtype(), weight.device())?;
        let gamma = Tensor::zeros(weight.dims()[0], weight.dtype(), weight.device())?;

        Ok(Self {
            num_groups: cfg.num_groups,
            weight,
            layer_norm,
            bias,
            eps: cfg.eps,
            q_b,
            beta,
            gamma,
        })
    }

    fn ste(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "ste");
        let _enter = span.enter();
        let binarized_x = sign(x)?;
        (binarized_x - x)?.detach() + x
    }

    fn binarize_weights_groupwise(&mut self) -> anyhow::Result<Tensor> {
        let group_size = self.weight.dims()[0] / self.num_groups;
        let binarized_weights = Tensor::zeros_like(&self.weight)?;
        let binarized_weights =
            (0..self.num_groups).fold(binarized_weights, |binarized_weights, i| {
                event!(tracing::Level::TRACE, "binarize-weights-groupwise-iter");
                let start_idx = i * group_size;
                let end_idx = (i + 1) * group_size;
                let weights_group = self.weight.narrow(0, start_idx, group_size).unwrap();
                let alpha_g = weights_group.mean_all().unwrap();
                self.beta = self
                    .beta
                    .slice_assign(&[start_idx..end_idx], &self.beta)
                    .unwrap();
                let binarized_weights = binarized_weights
                    .slice_assign(
                        &[start_idx..end_idx, 0..binarized_weights.dims()[1]],
                        &self
                            .ste(&(weights_group.broadcast_sub(&alpha_g).unwrap()))
                            .unwrap(),
                    )
                    .unwrap();
                binarized_weights
            });
        Ok(binarized_weights)
    }

    fn dequantize_activations(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "dequantize-activations");
        let _enter = span.enter();
        let x = (x.broadcast_mul(&self.gamma)?.broadcast_mul(&self.beta)? / self.q_b)?;
        Ok(x)
    }

    fn quantize_activations(&mut self, x: &Tensor) -> anyhow::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "quantize-activations");
        let _enter = span.enter();

        let group_size = x.dims()[0] / self.num_groups;
        let quantized_x = Tensor::zeros_like(&x)?;

        let quantized_x = (0..self.num_groups).fold(quantized_x, |quantized_x: Tensor, i| {
            let start_idx = i * group_size;
            let end_idx = (i + 1) * group_size;

            let activation_group = x.narrow(0, start_idx, group_size).unwrap();

            // torch.max gets max across all elements, candle doesn't have this so we flatten then call max on 0th dim
            let gamma_g = activation_group
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap();

            // We now need to expand gamma_g to the same shape as the slice we are assigning
            let gamma_g = gamma_g.expand(&[group_size]).unwrap();

            self.gamma = self
                .gamma
                .slice_assign(&[start_idx..end_idx], &gamma_g)
                .unwrap();

            let quantized_x_group = {
                let clamp_min = -self.q_b + self.eps as f64;
                let clamp_max = self.q_b - self.eps as f64;
                let x = (activation_group * self.q_b).unwrap();
                let x = x
                    .broadcast_div(
                        &gamma_g
                            .broadcast_add(&Tensor::new(self.eps, quantized_x.device()).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
                let x = x.clamp(clamp_min, clamp_max).unwrap();
                x
            };

            quantized_x
                .slice_assign(
                    &[
                        start_idx..end_idx,
                        0..quantized_x.dims()[1],
                        0..quantized_x.dims()[2],
                    ],
                    &quantized_x_group,
                )
                .unwrap()
        });
        Ok(quantized_x)
    }

    pub fn forward(&mut self, x: &Tensor) -> anyhow::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "bit-linear");
        let _enter = span.enter();

        // normalize input
        let x = self.layer_norm.forward(x)?;

        // binarize weights and quantize activations
        let binarized_weights = self.binarize_weights_groupwise()?;

        // quantize activations
        let x_quant = self.quantize_activations(&x)?;

        // perform linear transformation
        let output =
            Linear::from_weights(binarized_weights, self.bias.clone()).forward(&x_quant)?;

        // dequantize activations
        let output = self.dequantize_activations(&output)?;

        Ok(output)
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
        let mut bl = Bitlinear::load(
            BitlinearCfg {
                in_features,
                out_features,
                num_groups: 1,
                b: 8,
                eps: 1e-6,
                bias: true,
            },
            vb,
        )?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (1, 64), &device.clone())?;
        let output = bl.forward(&input)?;
        assert_eq!(output.shape().dims2()?, (64, 64));
        Ok(())
    }
}
