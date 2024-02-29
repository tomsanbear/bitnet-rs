use crate::bitffn::BitFeedForward;
use crate::multi_head_attention::MultiHeadAttention;
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

pub struct Transformer {
    dim: usize,
    heads: usize,
    depth: usize,
    ff_mult: usize,
    layers: Vec<MultiHeadAttention>,
    ffn_layers: Vec<BitFeedForward>,
}

impl Transformer {
    pub fn new(
        dim: usize,
        heads: usize,
        depth: usize,
        ff_mult: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut ffn_layers = Vec::new();

        // Layers are multihead attention blocks
        for _ in 0..depth {
            layers.push(MultiHeadAttention::load(dim, heads, vb.clone())?);
            ffn_layers.push(BitFeedForward::load(dim, ff_mult, vb.device())?);
        }

        Ok(Self {
            dim,
            heads,
            depth,
            ff_mult,
            layers,
            ffn_layers,
        })
    }
}

impl Module for Transformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

#[cfg(test)]
mod transformer_tests {
    use candle_core::{Device, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let tr = super::Transformer::new(512, 8, 6, 4, vb)?;
        assert_eq!(tr.dim, 512);
        assert_eq!(tr.heads, 8);
        assert_eq!(tr.depth, 6);
        assert_eq!(tr.ff_mult, 4);
        assert_eq!(tr.layers.len(), 6);
        assert_eq!(tr.ffn_layers.len(), 6);
        Ok(())
    }

    #[test]
    fn it_forwards() -> Result<()> {
        let vb = VarBuilder::zeros(candle_core::DType::F64, &Device::Cpu);
        let tr = super::Transformer::new(512, 8, 6, 4, vb)?;
        let x = Tensor::zeros(&[1, 512], candle_core::DType::F64, &Device::Cpu)?;
        let y = tr.forward(&x)?;
        assert_eq!(x.shape(), y.shape());
        Ok(())
    }
}
