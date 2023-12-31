use crate::error::Result;
use crate::loss::LossFn;
use crate::optimizer::{Optimizer, Optimizers};
use candle_core::{Device, Module, Tensor};
use candle_nn::VarMap;

mod sequential;

pub trait ModelBuilder {
    fn compile(
        &self,
        variables: &VarMap,
        device: &Device,
        loss_fn: LossFn,
        optimizer: Optimizers,
    ) -> Result<Box<dyn Model>>;
}

pub trait Model: Module {
    fn variables(&self) -> VarMap;

    fn loss(&self) -> LossFn;

    fn optimizer(&self) -> Result<Box<dyn Optimizer>>;

    fn fit(&self, x: &Tensor, y: &Tensor, epochs: usize) -> Result<TrainOutput> {
        let mut optimizer = self.optimizer()?;
        let mut sum_loss = 0_f32;

        for _step in 0..epochs {
            let ys = self.forward(&x)?;
            let loss = ys.sub(&y)?.sqr()?.sum_all()?;

            optimizer.backward_step(&loss)?;

            sum_loss += loss.to_vec0::<f32>()?;
        }

        Ok(TrainOutput { loss: sum_loss })
    }
}

pub struct TrainOutput {
    loss: f32,
}
