use crate::error::Result;
use crate::variables::Variable;
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

fn create_variable(
    name: &str,
    shape: &Shape,
    initializer: Option<Init>,
    dtype: DType,
    trainable: bool,
    device: &Device,
) -> Result<Variable> {
    let initializer = initializer.unwrap_or(
        // todo Should this be Glorot?
        if dtype.is_float() {
            DEFAULT_KAIMING_UNIFORM
        } else {
            ZERO
        },
    );

    Ok(Variable::new(
        initializer,
        shape,
        dtype,
        trainable,
        name.to_string(),
        device,
    )?)
}
