use crate::error::Result;
use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var};
use candle_nn::optim::{AdamW as AdamWInternal, SGD as SGDInternal};
use candle_nn::{Optimizer as OptimizerInternal, ParamsAdamW};

pub trait OptimizerBuilder {
    fn build(&self, vars: Vec<Var>) -> candle_core::Result<Box<dyn Optimizer>>;
}

pub enum Optimizers {
    AdamWDefault,
    AdamW(ParamsAdamW),
    Sgd(f64),
}

impl OptimizerBuilder for Optimizers {
    fn build(&self, vars: Vec<Var>) -> candle_core::Result<Box<dyn Optimizer>> {
        let result = match self {
            Self::AdamWDefault => {
                Box::new(AdamW::new(vars, ParamsAdamW::default())?) as Box<dyn Optimizer>
            }
            Self::Sgd(learning_rate) => {
                Box::new(Sgd::new(vars, learning_rate.clone())?) as Box<dyn Optimizer>
            }
            Self::AdamW(params) => {
                Box::new(AdamW::new(vars, params.clone())?) as Box<dyn Optimizer>
            }
        };

        Ok(result)
    }
}

pub trait Optimizer {
    fn step(&mut self, grads: &GradStore) -> Result<()>;

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }
}

pub struct AdamW {
    inner: AdamWInternal,
}

impl AdamW {
    fn new(vars: Vec<Var>, adam_w: ParamsAdamW) -> Result<Self> {
        Ok(AdamW {
            inner: AdamWInternal::new(vars, adam_w)?,
        })
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        Ok(self.inner.step(grads)?)
    }
}

pub struct Sgd {
    inner: SGDInternal,
}

impl Sgd {
    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        Ok(Sgd {
            inner: SGDInternal::new(vars, learning_rate)?,
        })
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.inner.step(grads)
    }
}
