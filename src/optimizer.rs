use candle_core::{Tensor, Var};
use tracing::instrument;

use candle_core::Result;

/// The interface optimizers should implement.
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let device = self.vars[0].var.device();
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        let beta1 = Tensor::new(beta1 as f32, &device)?;
        let beta2 = Tensor::new(beta2 as f32, &device)?;
        let lambda = Tensor::new((1f64 - lr_lambda) as f32, &device)?;
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.broadcast_mul(&beta1.clone()))?
                    + (g.broadcast_mul(&(1.0 - beta1.clone())?))?)?;
                let next_v = (v.broadcast_mul(&beta2.clone())?
                    + (g.sqr()?.broadcast_mul(&(1.0 - beta2.clone())?))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.broadcast_mul(&lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

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
