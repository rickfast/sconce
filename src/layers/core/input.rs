
use crate::builder_field;
use crate::layers::{Layer, LayerBuilder};
use candle_core::{DType, Module, Shape, Tensor};
use candle_nn::VarBuilder;

#[derive(Clone)]
pub struct Input {
    shape: Shape,
    batch_size: Option<usize>,
    dtype: DType,
}

impl Input {
    fn new(shape: Shape, dtype: DType) -> crate::error::Result<Self> {
        Ok(Self {
            shape,
            batch_size: None,
            dtype,
        })
    }

    builder_field!(batch_size, Option<usize>);
}

impl LayerBuilder for Input {
    fn build(&self, input_shape: &Shape, _vs: &VarBuilder) -> crate::error::Result<Box<dyn Layer>> {
        Ok(
            Box::new(InputLayer {
                shape: input_shape.clone(),
                batch_size: self.batch_size,
                dtype: self.dtype,
            })
        )
    }
}

struct InputLayer {
    shape: Shape,
    batch_size: Option<usize>,
    dtype: DType,
}

impl Layer for InputLayer {
    fn name(&self) -> String {
        todo!()
    }
}

impl Module for InputLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(xs.clone())
    }
}
