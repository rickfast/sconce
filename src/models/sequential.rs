use std::cell::RefCell;
use std::rc::Rc;
use crate::layers::dense::Dense;
use crate::layers::{Layer, LayerBuilder};
use crate::models::{Model, ModelBuilder};
use crate::error::Result;
use candle_core::{Device, DType, Module, Shape, Tensor, Var};
use candle_nn::init::ZERO;
use candle_nn::{VarMap};
use crate::loss::LossFn;
use crate::optimizer::{Optimizer, OptimizerBuilder, Optimizers};

struct Sequential<'a> {
    layers: Vec<Box<&'a mut dyn LayerBuilder>>,
}

impl<'a> Sequential<'a> {
    fn new() -> Self {
        Sequential { layers: vec![] }
    }
    fn add_layer<L: LayerBuilder + 'a>(&'a mut self, layer: &'a mut L) -> &'a mut Self {
        self.layers.push(Box::new(layer));

        self
    }
}

impl ModelBuilder for Sequential<'_> {
    type M = SequentialModel;

    fn compile(
        &self,
        varmap: &VarMap,
        device: &Device,
        loss_fn: LossFn,
        optimizers: Optimizers
    ) -> Result<Box<dyn Model>> {
        let input_shape = Shape::from_dims(&[1, 1]);
        let layers: Vec<Box<dyn Layer>> = self
            .layers
            .iter()
            .map(|layer| layer.build(&input_shape.clone().into(), varmap, device).unwrap())
            .collect();

        Ok(Box::new(SequentialModel { layers, variables: varmap.clone(), loss: loss_fn, optimizers }))
    }
}

struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
    variables: VarMap,
    loss: LossFn,
    optimizers: Optimizers
}

impl Module for SequentialModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(self
            .layers
            .iter()
            .fold(xs.clone(), |x, layer| layer.forward(&x).unwrap()))
    }
}

impl Model for SequentialModel {
    fn variables(&self) -> VarMap {
        self.variables.clone()
    }

    fn loss(&self) -> LossFn {
        self.loss
    }

    fn optimizer(&self) -> Result<Box<dyn Optimizer>> {
        let vars = self.variables.all_vars();
        self.optimizers.build(vars)
    }
}

fn do_it() {
    let _sequential = Sequential::new()
        .add_layer(Dense::new(2)
            .kernel_initializer(Some(ZERO)));
}
