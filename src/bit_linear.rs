use crate::utils_tensor::sign;
use candle_core::{Tensor, D};
use candle_nn::{layer_norm, Init, LayerNorm, LayerNormConfig, Module, VarBuilder};
use candle_transformers::models::with_tracing::Linear;
use tracing::{event, span};

pub struct Bitlinear {
    num_groups: usize,
    weight: Tensor,
    bias: Option<Tensor>,
    layer_norm: LayerNorm,
    b: i32,
    eps: f32,
    span: tracing::Span,
}

impl Bitlinear {
    pub fn load(
        in_features: usize,
        out_features: usize,
        num_groups: usize,
        b: i32,
        eps: f32,
        bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "bit-linear");
        let weight = vb.get_with_hints(
            (out_features, in_features),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let layer_norm = layer_norm(
            in_features,
            LayerNormConfig {
                eps: eps.into(),
                ..LayerNormConfig::default()
            },
            vb.pp("layer_norm"),
        )?;
        let bias = match bias {
            true => Some(vb.get_with_hints(out_features, "bias", Init::Const(0.0))?),
            false => None,
        };
        Ok(Self {
            span,
            num_groups,
            weight,
            layer_norm,
            b,
            bias,
            eps,
        })
    }

    fn ste(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "ste");
        let _enter = span.enter();
        let binarized_x = sign(x)?;
        let binarized_x = binarized_x.sub(x)?.detach().add(x)?;
        Ok(binarized_x)
    }

    fn binarize_weights_groupwise(&self) -> candle_core::Result<Tensor> {
        let span = tracing::span!(tracing::Level::TRACE, "binarize-weights-groupwise");
        let _enter = span.enter();
        let group_size = self.weight.dims()[0] / self.num_groups;
        let mut bin_weights = Vec::with_capacity(self.num_groups);
        for i in 0..self.num_groups {
            event!(tracing::Level::TRACE, "binarize-weights-groupwise-iter");
            let d0_start_idx = i * group_size;
            let weight_group = self.weight.narrow(0, d0_start_idx, group_size)?;
            let alpha_g = weight_group.mean_all()?;
            let ste_result = self.ste(&weight_group.broadcast_sub(&alpha_g)?)?;
            bin_weights.push(ste_result);
        }
        let bin_weights = bin_weights.into_boxed_slice();
        let output = Tensor::cat(&bin_weights, 1).unwrap();
        Ok(output)
    }

    fn dequantize_activations(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "dequantize-activations");
        let _enter = span.enter();

        let q_b = 2f32.powi(self.b);
        let q_b_t = Tensor::new(q_b, x.device())?;
        let group_size = x.dims()[1] / self.num_groups;

        let mut grouped_results = Vec::with_capacity(self.num_groups);
        for g in 0..self.num_groups {
            event!(tracing::Level::TRACE, "dequantize-activations-iter");

            let start_idx = g * group_size;
            let quantized_group = x.narrow(1, start_idx, group_size)?;
            let gamma_g = quantized_group.abs()?.max_keepdim(D::Minus1)?;
            let dequantized_x = quantized_group
                .broadcast_mul(&gamma_g)?
                .broadcast_div(&q_b_t)?;
            grouped_results.push(dequantized_x);
        }
        let quantized_x = grouped_results.into_boxed_slice();
        let output = Tensor::cat(&quantized_x, 1).unwrap();
        Ok(output)
    }

    fn quantize_activations(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let span = span!(tracing::Level::TRACE, "quantize-activations");
        let _enter = span.enter();

        let q_b = 2f32.powi(self.b);
        let q_b_t = Tensor::new(q_b, x.device())?;
        let group_size = x.dims()[1] / self.num_groups;

        let mut grouped_results = Vec::with_capacity(self.num_groups);
        for g in 0..self.num_groups {
            event!(tracing::Level::TRACE, "binarize-weights-groupwise-iter");

            let start_idx = g * group_size;
            let activation_group = x.narrow(1, start_idx, group_size)?;
            let gamma_g = activation_group.abs()?.max_keepdim(D::Minus1)?;
            let quantized_x = activation_group.broadcast_mul(&q_b_t)?;
            let quantized_x = quantized_x.broadcast_div(
                &(gamma_g.broadcast_add(&Tensor::new(self.eps, quantized_x.device())?)?),
            )?;
            let quantized_x = quantized_x.clamp(-q_b + self.eps, q_b - self.eps)?;
            grouped_results.push(quantized_x);
        }
        let quantized_x = grouped_results.into_boxed_slice();
        let output = Tensor::cat(&quantized_x, 1).unwrap();
        Ok(output)
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();

        // normalize input
        let x = self.layer_norm.forward(x)?;

        // binarize weights and quantize activations
        let binarized_weights = self.binarize_weights_groupwise()?;

        // perform linear transformation
        let output = Linear::from_weights(binarized_weights, self.bias.clone()).forward(&x)?;

        // quantize activations
        let output = self.quantize_activations(&output)?;

        // dequantize activations
        let output = self.dequantize_activations(&output)?;

        Ok(output)
    }
}

#[cfg(test)]
mod bitlinear_tests {
    use super::Bitlinear;
    use crate::utils_tensor::device;
    use candle_core::{DType, Module, Result, Tensor};
    use candle_nn::var_builder::VarBuilderArgs;

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let device = device(true).unwrap();
        let vb = VarBuilderArgs::zeros(DType::F32, &device.clone());
        let in_features = 64;
        let out_features = 64;
        let bl = Bitlinear::load(in_features, out_features, 1, 8, 1e-6, true, vb)?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (1, 64), &device.clone())?;
        let output = bl.forward(&input)?;
        assert_eq!(output.shape().dims2()?, (1, 64));
        Ok(())
    }
}
