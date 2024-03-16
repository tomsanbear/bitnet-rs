use candle_core::Tensor;
use tracing::instrument;

#[derive(Debug)]
pub struct DropoutCfg {
    pub p: f32,
    pub is_training: bool,
}

#[derive(Debug)]
pub struct Dropout {
    drop_p: f32,
    is_training: bool,
}

impl Dropout {
    #[instrument]
    pub fn load(cfg: DropoutCfg) -> anyhow::Result<Self> {
        Ok(Self {
            drop_p: cfg.p,
            is_training: cfg.is_training,
        })
    }

    #[instrument]
    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        if !self.is_training {
            return Ok(x.clone());
        }

        if !(0. ..1.).contains(&self.drop_p) {
            anyhow::bail!(
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

        Ok((x * mask)?)
    }
}
