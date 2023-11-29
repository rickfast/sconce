use crate::error::Result;
use candle_core::{Device, Module, Shape, Var};

use candle_nn::{VarBuilder, VarMap};

pub mod dense;
pub mod input;

pub trait Layer: Module {
    fn name(&self) -> String;
}

pub trait LayerBuilder {
    fn build(&self, input_shape: &Shape, varmap: &VarMap, device: &Device) -> Result<Box<dyn Layer>>;
}
