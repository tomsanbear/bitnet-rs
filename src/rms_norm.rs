use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use tracing::instrument;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
}

impl RmsNorm {
    pub fn load(rms_norm_eps: f64, size: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::rms_norm(size, rms_norm_eps, vb)?;
        Ok(Self { inner })
    }
}

impl Module for RmsNorm {
    #[instrument]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

#[cfg(test)]
mod rmsnorm_tests {

    use super::RmsNorm;
    use candle_core::{DType, Device, Module, Result, Tensor};
    use candle_nn::VarBuilder;

    #[test]
    fn it_loads() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F64, &Device::Cpu);
        RmsNorm::load(1e-6, 512, vb)?;
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let rmsnorm = RmsNorm::load(1e-6, 512, vb)?;
        let input = Tensor::ones((1, 512), DType::F32, &Device::Cpu)?;
        let output = rmsnorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 512]);
        Ok(())
    }
}
