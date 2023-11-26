use crate::error::Result;
use candle_core::{Module, Shape};

use candle_nn::VarBuilder;

pub mod core;

pub trait Layer: Module {
    fn name(&self) -> String;
}

pub trait LayerBuilder {
    fn build(&self, input_shape: &Shape, vs: &VarBuilder) -> Result<Box<dyn Layer>>;
}
