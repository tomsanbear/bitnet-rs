use crate::utils_tensor::{absmean_quantize_weights, sign};
use candle_core::Tensor;
use candle_nn::{layer_norm, Init, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};

pub struct Bitlinear {
    num_groups: usize,
    weight: Tensor,
    layer_norm: LayerNorm,
}

impl Bitlinear {
    pub fn load(
        in_features: usize,
        out_features: usize,
        num_groups: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let weight = vb.get_with_hints(
            (out_features, in_features),
            "weights",
            Init::Randn {
                mean: 0.0,
                stdev: 1.0,
            },
        )?;
        let layer_norm = layer_norm(
            in_features,
            LayerNormConfig {
                ..LayerNormConfig::default()
            },
            vb.pp("layer_norm"),
        )?;
        Ok(Self {
            num_groups,
            weight,
            layer_norm,
        })
    }

    fn ste(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // binarized_x = torch.sign(x)
        let binarized_x = sign(x)?;
        let binarized_x = binarized_x.sub(x)?.detach().add(x)?;
        Ok(binarized_x)
    }

    fn binarize_weights_groupwise(&self) -> candle_core::Result<Tensor> {
        let group_size = self.weight.dims()[0] / self.num_groups;
        let mut bin_weights = self.weight.zeros_like()?;
        for i in 0..self.num_groups {
            let d0_start_idx = i * group_size;
            let d0_end_idx = (i + 1) * group_size;
            let d1_end_idx = bin_weights.dims()[1];
            let weight_group = self.weight.narrow(0, d0_start_idx, group_size)?;
            let alpha_g = weight_group.mean_all()?;
            let ste_result = self.ste(&weight_group.broadcast_sub(&alpha_g)?)?;
            bin_weights = bin_weights
                .slice_assign(&[d0_start_idx..d0_end_idx, (0..d1_end_idx)], &ste_result)?;
        }
        Ok(bin_weights)
    }
}

impl Module for Bitlinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // normalize input
        let x = self.layer_norm.forward(x)?;

        // binarize weights and quantize activations
        let binarized_weights = self.binarize_weights_groupwise()?;

        // perform linear transformation
        let output = Linear::new(binarized_weights, None).forward(&x)?;

        // quantize activations
        let output = absmean_quantize_weights(&output)?;

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
        let in_features = 768;
        let out_features = 32;
        let bl = Bitlinear::load(in_features, out_features, 2, vb)?;
        let input: Tensor = Tensor::randn(0.0f32, 1.0f32, (4, 64, 1024), &device.clone())?;
        let output = bl.forward(&input)?;
        assert_eq!(output.shape().dims2()?, (5, 5));
        Ok(())
    }
}
