use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::LayerNorm;

pub struct RMSNorm {
    scale: Tensor,
    gamma: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, device: &Device) -> Result<Self> {
        let scale = Tensor::from_vec(vec![(dim as f64).powf(0.5)], 1, device)?.pad_with_same(
            0,
            0,
            dim - 1,
        )?;
        let gamma = Tensor::ones(dim, DType::F64, device)?;
        Ok(Self {
            scale,
            gamma,
            eps: 1e-8,
        })
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let layer_weights = x.ones_like()?;
        println!("{:?}", layer_weights.shape());
        let l2norm = LayerNorm::new_no_bias(layer_weights, self.eps).forward(x)?;
        println!("{:?}", l2norm.shape());
        let output = l2norm.mul(&self.gamma.unsqueeze(0)?)?;
        println!("{:?}", output.shape());
        let output = output.mul(&self.scale.unsqueeze(0)?)?;
        Ok(output)
    }
}

#[cfg(test)]
mod rmsnorm_tests {
    use super::RMSNorm;
    use candle_core::{DType, Device, Module, Result, Tensor};

    #[test]
    fn it_loads() -> Result<()> {
        RMSNorm::new(512, &Device::Cpu)?;
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let rmsnorm = RMSNorm::new(512, &Device::Cpu)?;
        let input = Tensor::ones((1, 512), DType::F64, &Device::Cpu)?;
        let output = rmsnorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 512]);
        Ok(())
    }
}
