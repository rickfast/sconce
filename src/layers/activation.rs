use candle_core::{Device, Module, Shape, Tensor};
use candle_nn::VarMap;
use crate::layers::{Layer, LayerBuilder, LayerConfiguration};

#[derive(Clone)]
pub struct Activation {
    activation: candle_nn::Activation
}

impl Activation {
    pub fn new(activation: candle_nn::Activation) -> Self {
        Activation {
            activation
        }
    }
}

impl LayerBuilder for Activation {
    fn build(&mut self) -> Self {
        self.clone()
    }
}

impl LayerConfiguration for Activation {
    fn compile(&self, input_shape: &Shape, variables: &VarMap, device: &Device) -> crate::error::Result<Box<dyn Layer>> {
        Ok(Box::new(ActivationLayer::new(self.activation)))
    }

    fn input_shape(&self) -> Option<Shape> {
        None
    }
}

pub struct ActivationLayer {
    activation: candle_nn::Activation
}

impl ActivationLayer {
    fn new(activation: candle_nn::Activation) -> Self {
        ActivationLayer {
            activation
        }
    }
}

impl Layer for ActivationLayer {
    fn name(&self) -> String {
        todo!()
    }
}

impl Module for ActivationLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.activation.forward(xs)
    }
}