use crate::error::Result;
use crate::layers::dense::Dense;
use crate::layers::{Layer, LayerBuilder, LayerConfiguration};
use crate::loss::LossFn;
use crate::models::{Model, ModelBuilder};
use crate::optimizer::{Optimizer, OptimizerBuilder, Optimizers};

use candle_core::{Device, Module, Tensor};
use candle_nn::Activation::Relu;
use candle_nn::init::ZERO;
use candle_nn::loss::nll;
use candle_nn::VarMap;
use crate::layers::activation::Activation;

struct Sequential {
    layers: Vec<Box<dyn LayerConfiguration>>,
}

pub(super) enum LayerOrModel {
    Layer(Box<dyn LayerConfiguration>),
    Model(Box<dyn ModelBuilder>),
}

impl<'a> Sequential {
    fn new() -> Self {
        Sequential { layers: vec![] }
    }
    fn add_layer<L: LayerConfiguration + Clone + 'a + 'static>(
        &'a mut self,
        layer: &'a L,
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
            .map(|layer| layer.compile(&input_shape, variables, device).unwrap())
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

fn train_example() -> Result<()> {
    let device = &Device::Cpu;
    let variables = VarMap::new();
    let model = Sequential::new()
        .add_layer(&Dense::new(2).kernel_initializer(Some(ZERO)).build())
        .add_layer(&Activation::new(Relu).build())
        .compile(&variables, &Device::Cpu, nll, Optimizers::AdamWDefault)?;
    let x = &Tensor::new(&[1.], &device)?;
    let y = &Tensor::new(&[1.], &device)?;

    let output = model.fit(x, y, 10)?;

    println!("Loss: {}", output.loss);

    Ok(())
}
