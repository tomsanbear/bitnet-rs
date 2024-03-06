use crate::bit_linear::Bitlinear;
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, seq, Dropout, LayerNormConfig, Sequential, VarBuilder};

pub struct BitFeedForwardCfg {
    pub dim: usize,
    pub ff_mult: usize,
    pub dropout: f32,
    pub train: bool,
    pub eps: f64,
}

pub struct BitFeedForward {
    layer: Sequential,
}

impl BitFeedForward {
    pub fn load(cfg: BitFeedForwardCfg, vb: VarBuilder) -> candle_core::Result<Self> {
        let inner_dim = cfg.dim * cfg.ff_mult;
        let activation = Tensor::gelu;
        let dropout = Dropout::new(cfg.dropout);
        let norm = layer_norm(
            inner_dim,
            LayerNormConfig {
                eps: cfg.eps,
                ..LayerNormConfig::default()
            },
            vb.pp("ffn_layer_norm"),
        )?;
        let layer = seq()
            .add(Bitlinear::load(
                cfg.dim,
                inner_dim,
                1,
                vb.pp("ffn_bitlinear_0"),
            )?)
            .add(activation)
            .add_fn(move |x| norm.forward(x))
            .add_fn(move |x| dropout.forward(x, cfg.train))
            .add(Bitlinear::load(
                inner_dim,
                cfg.dim,
                1,
                vb.pp("ffn_bitlinear_1"),
            )?);
        Ok(Self { layer })
    }
}

impl Module for BitFeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.layer.forward(x)
    }
}

#[cfg(test)]
mod bitffn_tests {
    use super::BitFeedForward;
    use crate::{bit_ffn::BitFeedForwardCfg, utils_tensor::device};
    use candle_core::{DType, Device, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_applies_forward_pass_dim_2() -> Result<()> {
        let device: Device = device(true).unwrap();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let dim = 128;
        let input: Tensor = Tensor::randn(0f32, 1.0, (10, dim), &device)?;
        let bff = BitFeedForward::load(
            BitFeedForwardCfg {
                dim,
                ff_mult: 4,
                dropout: 0.1,
                train: true,
                eps: 1e-6,
            },
            vb,
        )?;
        let output = bff.forward(&input).unwrap();
        let output_shape = output.shape().dims2()?;
        assert_eq!(output_shape, (10, dim));
        Ok(())
    }
}
