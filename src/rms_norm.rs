use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    pub fn load(rms_norm_eps: f32, size: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, rms_norm_eps.into(), vb)?;
        Ok(Self { inner, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
        RmsNorm::load(1e-6f32, 512, vb)?;
        Ok(())
    }

    #[test]
    fn it_applies_forward_pass() -> Result<()> {
        let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
        let rmsnorm = RmsNorm::load(1e-6f32, 512, vb)?;
        let input = Tensor::ones((1, 512), DType::F32, &Device::Cpu)?;
        let output = rmsnorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 512]);
        Ok(())
    }
}
