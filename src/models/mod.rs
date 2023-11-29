use std::cell::RefCell;
use std::rc::Rc;
use std::borrow::BorrowMut;
use crate::error::Result;
use crate::loss::LossFn;
use crate::optimizer::{Optimizer, OptimizerBuilder, Optimizers};
use candle_core::{Device, Module, Tensor, Var};
use candle_nn::VarMap;

mod sequential;

trait ModelBuilder {
    type M: Model;
    fn compile(
        &self,
        varmap: &VarMap,
        device: &Device,
        loss_fn: LossFn,
        optimizers: Optimizers
    ) -> Result<Box<dyn Model>>;
}

trait Model: Module {
    fn variables(&self) -> VarMap;

    fn loss(&self) -> LossFn;

    fn optimizer(&self) -> Result<Box<dyn Optimizer>>;

    fn fit(&self, x: &Tensor, y: &Tensor, epochs: usize) -> Result<TrainOutput> {
        let mut optimizer = self.optimizer()?;

        for step in 0..epochs {
            let ys = self.forward(&x)?;
            let loss = ys.sub(&y)?.sqr()?.sum_all()?;

            optimizer.backward_step(&loss)?;

            println!("{step} {}", loss.to_vec0::<f32>()?);
        }

        todo!()
    }
}

pub struct TrainOutput {
    y: Tensor,
    loss: f32,
}