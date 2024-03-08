use candle_core::Tensor;
use candle_nn::Module;

pub struct DropoutCfg {
    pub p: f32,
    pub is_training: bool,
}

// Re-implementation of the dropout mechanism since the default one from candle_nn is full of unsupported features
pub struct Dropout {
    span: tracing::Span,
    drop_p: f32,
    is_training: bool,
}

impl Dropout {
    pub fn load(cfg: DropoutCfg) -> anyhow::Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "dropout");
        Ok(Self {
            span,
            drop_p: cfg.p,
            is_training: cfg.is_training,
        })
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        if !self.is_training {
            return Ok(x.clone());
        }

        if !(0. ..1.).contains(&self.drop_p) {
            candle_core::bail!(
                "dropout probability has to be in [0, 1), got {:?}",
                self.drop_p
            )
        }
        let rand = Tensor::rand(0f32, 1f32, x.shape(), x.device())?;
        let scale = 1.0 / (1.0 - self.drop_p as f64);
        let drop_p = Tensor::new(self.drop_p, x.device())?.broadcast_as(x.shape())?;
        // Metal doesn't support contiguous affine operation so we need to cast to f32
        let mask = match x.device() {
            candle_core::Device::Metal(_) => {
                (rand.ge(&drop_p)?.to_dtype(candle_core::DType::F32)? * scale)?.to_dtype(x.dtype())
            }
            _ => (rand.ge(&drop_p)? * scale)?.to_dtype(x.dtype()),
        }?;

        x * mask
    }
}
