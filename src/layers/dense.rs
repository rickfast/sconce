use candle_core::{DType, Device, Module, Shape, Tensor};
use candle_nn::init::{DEFAULT_KAIMING_UNIFORM, ZERO};
use candle_nn::{Activation, Init, VarBuilder, VarMap};

use crate::builder_field;
use crate::builder_option_field;
use crate::error::Result;
use crate::layers::{Layer, LayerBuilder, LayerConfiguration};
use crate::regularizer::Regularizer;

#[derive(Clone)]
pub struct Dense {
    units: usize,
    activation: Option<Activation>,
    use_bias: bool,
    kernel_initializer: Option<Init>,
    bias_initializer: Option<Init>,
    activity_regularizer: Option<Regularizer>,
    bias_regularizer: Option<Regularizer>,
}

impl Dense {
    pub fn new(units: usize) -> Self {
        Self {
            units,
            activation: None,
            use_bias: true,
            kernel_initializer: None,
            bias_initializer: None,
            activity_regularizer: None,
            bias_regularizer: None,
        }
    }

    builder_option_field!(activation, Activation);
    builder_field!(use_bias, bool);
    builder_option_field!(kernel_initializer, Init);
    builder_option_field!(bias_initializer, Init);
    builder_option_field!(activity_regularizer, Regularizer);
    builder_option_field!(bias_regularizer, Regularizer);
}

impl LayerBuilder for Dense {
    fn build(&mut self) -> Dense {
        self.clone()
    }
}

impl LayerConfiguration for Dense {
    fn compile(
        &self,
        input_shape: &Shape,
        varmap: &VarMap,
        device: &Device,
    ) -> Result<Box<dyn Layer>> {
        let input_dim = input_shape.dims().last().unwrap();
        let vs = VarBuilder::from_varmap(varmap, DType::F32, device);
        let kernel = vs.get_with_hints(
            &[*input_dim, self.units],
            "kernel",
            self.kernel_initializer.unwrap_or(ZERO),
        )?;
        let bias = match self.bias_initializer {
            Some(_init) => Some(vs.get_with_hints(
                &[self.units],
                "bias",
                self.bias_initializer.unwrap_or(DEFAULT_KAIMING_UNIFORM),
            )?),
            None => None,
        };

        Ok(Box::new(DenseLayer {
            units: self.units,
            activation: self.activation,
            kernel,
            bias,
        }))
    }
}

fn dense(units: usize) -> Dense {
    Dense::new(units).build()
}

pub(crate) struct DenseLayer {
    units: usize,
    activation: Option<Activation>,
    kernel: Tensor,
    bias: Option<Tensor>,
}

impl Module for DenseLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mut result = xs.matmul(&self.kernel)?;

        result = match &self.bias {
            Some(bias) => result.add(&bias)?,
            None => result,
        };

        result = match &self.activation {
            Some(activation) => activation.forward(&result)?,
            None => result,
        };

        Ok(result)
    }
}

impl Layer for DenseLayer {
    fn name(&self) -> String {
        todo!()
    }
}
