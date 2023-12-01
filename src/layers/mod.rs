use crate::error::Result;
use candle_core::{Device, Module, Shape};

use candle_nn::VarMap;

pub mod dense;
pub mod input;
pub mod activation;

pub trait Layer: Module {
    fn name(&self) -> String;
}

pub trait LayerBuilder {
    fn build(&mut self) -> Self;
}

pub trait LayerConfiguration {
    fn compile(
        &self,
        input_shape: &Shape,
        variables: &VarMap,
        device: &Device,
    ) -> Result<Box<dyn Layer>>;

    fn input_shape(&self) -> Option<Shape> {
        None
    }
}
