use crate::utils_tensor::sign;
use anyhow::Ok;
use candle_core::{Tensor, D};
use candle_nn::{layer_norm, Init, LayerNorm, LayerNormConfig, Module, VarBuilder};
use candle_transformers::models::with_tracing::Linear;
use tracing::instrument;

#[derive(Debug, Clone, Copy)]
pub struct BitlinearCfg {
    pub in_features: usize,
    pub out_features: usize,
    pub num_groups: usize,
    pub b: i32,
    pub eps: f32,
    pub bias: bool,
}

#[derive(Debug)]
pub struct Bitlinear {
    num_groups: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    layer_norm: LayerNorm,
    eps: f64,
    q_b: f64,
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
        let bias = match cfg.bias {
            true => Some(vb.get_with_hints(cfg.out_features, "bias", Init::Const(0.0))?),
            false => None,
        };
        let layer_norm = layer_norm(
            cfg.in_features,
            LayerNormConfig {
                eps: cfg.eps.into(),
                ..LayerNormConfig::default()
            },
            vb.pp("layer_norm"),
        )?;
        let q_b = 2f64.powi(cfg.b - 1);
        Ok(Self {
            num_groups: cfg.num_groups,
            weight,
            layer_norm,
            bias,
            eps: cfg.eps as f64,
            q_b,
        })
    }

    #[instrument]
    fn ste(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let binarized_x = sign(x)?;
        (binarized_x - x)?.detach() + x
    }

    #[instrument]
    fn binarize_weights_groupwise(&self) -> anyhow::Result<(Tensor, Tensor)> {
        /*
         * Note:
         * The original code uses slice assignment on a zeroed tensor to create the final tensor
         * We instead push the chunks into a vector then combine at the very end to avoid the need to call
         * slice_assign.
         */
        let mut binarized_weight_groups: Vec<Tensor> = Vec::with_capacity(self.num_groups);
        let mut beta_groups: Vec<Tensor> = Vec::with_capacity(self.num_groups);
        let group_size = self.weight.dims()[0] / self.num_groups;
        for i in 0..self.num_groups {
            let start_idx = i * group_size;
            let weights_group = self.weight.narrow(0, start_idx, group_size)?;
            let alpha_g = weights_group.mean_all()?;
            let beta = weights_group.abs()?.mean(D::Minus1)?.mean(D::Minus1)?;
            beta_groups.push(beta);
            let binarized_weights = self.ste(&(weights_group.broadcast_sub(&alpha_g)?))?;
            binarized_weight_groups.push(binarized_weights);
        }

        let binarized_weights = Tensor::cat(&binarized_weight_groups, D::Minus1)?;
        let beta = Tensor::cat(&beta_groups, D::Minus1)?;
        Ok((binarized_weights, beta))
    }

    #[instrument]
    fn dequantize_activations(
        &self,
        x: &Tensor,
        beta: &Tensor,
        gamma: &Tensor,
    ) -> anyhow::Result<Tensor> {
        Ok((x
            .broadcast_mul(
                &gamma
                    .unsqueeze(D::Minus1)
                    .unwrap()
                    .unsqueeze(D::Minus1)
                    .unwrap(),
            )?
            .broadcast_mul(beta)?
            / self.q_b)?)
    }

    #[instrument]
    fn quantize_activations(&self, x: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let mut quantized_x_groups: Vec<Tensor> = Vec::with_capacity(self.num_groups);
        let mut gamma_groups: Vec<Tensor> = Vec::with_capacity(self.num_groups);
        let group_size = x.dims()[0] / self.num_groups;
        for i in 0..self.num_groups {
            let start_idx = i * group_size;
            let activation_group = x.narrow(0, start_idx, group_size).unwrap();
            let gamma = activation_group.abs()?.max(D::Minus1)?.max(D::Minus1)?;
            let clamp_min = -self.q_b + self.eps;
            let clamp_max = self.q_b - self.eps;
            let x = (activation_group * self.q_b).unwrap();
            let x = x
                .broadcast_div(
                    &(gamma.clone() + self.eps)
                        .unwrap()
                        .unsqueeze(D::Minus1)
                        .unwrap()
                        .unsqueeze(D::Minus1)
                        .unwrap(),
                )
                .unwrap();
            let quantized_x = x.clamp(clamp_min, clamp_max).unwrap();
            quantized_x_groups.push(quantized_x);
            gamma_groups.push(gamma.clone());
        }
        let quantized_x = Tensor::cat(&quantized_x_groups, 0)?;
        let gamma = Tensor::cat(&gamma_groups, 0)?;
        Ok((quantized_x, gamma))
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        // normalize input
        let x = self.layer_norm.forward(x)?;

        // binarize weights and quantize activations
        let (binarized_weights, beta) = self.binarize_weights_groupwise()?;

        // quantize activations
        let (x_quantized, gamma) = self.quantize_activations(&x)?;

        // perform linear transformation
        let output = match &self.bias {
            Some(bias) => {
                Linear::from_weights(binarized_weights, Some(bias.clone())).forward(&x_quantized)?
            }
            None => Linear::from_weights(binarized_weights, None).forward(&x_quantized)?,
        };

        // dequantize activations
        let output = self.dequantize_activations(&output, &beta, &gamma)?;

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
        let bl = Bitlinear::load(
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
