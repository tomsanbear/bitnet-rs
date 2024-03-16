use candle_core::{Tensor, Var};
use candle_nn::{AdamW, Optimizer};
use tracing::span;

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

    pub fn backward_step(&mut self, loss: &Tensor) -> anyhow::Result<()> {
        let span = span!(tracing::Level::TRACE, "backward-step");
        let _enter = span.enter();
        Ok(self.inner.backward_step(loss)?)
    }
}
