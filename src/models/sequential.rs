use crate::error::Result;
use crate::layers::dense::Dense;
use crate::layers::{Layer, LayerBuilder};
use crate::loss::LossFn;
use crate::models::{Model, ModelBuilder};
use crate::optimizer::{Optimizer, OptimizerBuilder, Optimizers};
use anyhow::Error;
use candle_core::{Device, Module, Tensor};
use candle_nn::init::ZERO;
use candle_nn::VarMap;

struct Sequential {
    layers: Vec<Box<dyn LayerBuilder>>,
}

pub(super) enum LayerOrModel {
    Layer(Box<dyn LayerBuilder>),
    Model(Box<dyn ModelBuilder>),
}

impl<'a> Sequential {
    fn new() -> Self {
        Sequential { layers: vec![] }
    }
    fn add_layer<L: LayerBuilder + Clone + 'a + 'static>(
        &'a mut self,
        layer: &'a mut L,
    ) -> &'a mut Self {
        self.layers.push(Box::new(layer.clone()));

        self
    }
}

impl ModelBuilder for Sequential {
    fn compile(
        &self,
        variables: &VarMap,
        device: &Device,
        loss_fn: LossFn,
        optimizer: Optimizers,
    ) -> Result<Box<dyn Model>> {
        let input_layer = self
            .layers
            .first()
            .ok_or_else(|| crate::error::Error::EmptyModelArchitecture)?;
        let input_shape = input_layer
            .input_shape()
            .ok_or_else(|| crate::error::Error::InputShapeNotFound)?;
        let layers: Vec<Box<dyn Layer>> = self
            .layers
            .iter()
            .map(|layer| layer.build(&input_shape, variables, device).unwrap())
            .collect();

        Ok(Box::new(SequentialModel {
            layers,
            variables: variables.clone(),
            loss: loss_fn,
            optimizers: optimizer,
        }))
    }
}

struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
    variables: VarMap,
    loss: LossFn,
    optimizers: Optimizers,
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
    let model = Sequential::new().add_layer(Dense::new(2).kernel_initializer(Some(ZERO)));
}
