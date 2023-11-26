use crate::error::Result;
use candle_core::{DType, Device, Module, Shape, Tensor, Var, WithDType};
use candle_nn::init::{DEFAULT_KAIMING_UNIFORM, ZERO};
use candle_nn::{Init, VarBuilder};

mod core;

pub trait Layer: candle_core::Module {
    fn name(&self) -> String;
}

pub trait LayerBuilder<L: Layer> {
    fn build(&self, input_shape: &Shape, vs: &VarBuilder) -> Result<L>;
}