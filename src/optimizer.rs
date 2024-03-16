use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use tracing::instrument;

#[derive(Debug)]
pub struct BitnetOptimizer {
    inner: AdamW,
}

/// Wrapper around the AdamW optimizer
/// Includes additional tracing features and project defaults
impl BitnetOptimizer {
    pub fn load(vars: Vec<Var>, learning_rate: f64) -> anyhow::Result<Self> {
        let inner = AdamW::new_lr(vars, learning_rate)?;
        Ok(Self { inner })
    }

    #[instrument]
    pub fn backward_step(&mut self, loss: &Tensor) -> anyhow::Result<()> {
        Ok(self.inner.backward_step(loss)?)
    }
}
