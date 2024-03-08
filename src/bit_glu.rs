use crate::bit_linear::Bitlinear;
use candle_core::{Tensor, D};
use candle_nn::{Activation, Module, VarBuilder};

pub struct BitGLU {
    proj: Bitlinear,
    activation: Activation,
}

impl BitGLU {
    pub fn load(
        in_features: usize,
        out_features: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> anyhow::Result<Self> {
        let proj = Bitlinear::load(in_features, out_features * 4, 1, vb.pp("proj"))?;
        Ok(BitGLU { proj, activation })
    }
}

impl Module for BitGLU {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.proj.forward(x)?;
        let chunks = x.chunk(2, D::Minus1)?;
        let x = chunks.get(0).unwrap();
        let gate = chunks.get(0).unwrap();
        let output = x.mul(&self.activation.forward(&gate)?)?;
        Ok(output)
    }
}
