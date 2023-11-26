use candle_core::{Module, Shape, Tensor};
use candle_nn::init::DEFAULT_KAIMING_UNIFORM;
use candle_nn::{Activation, Init, VarBuilder};

use crate::builder_field;
use crate::error::Result;
use crate::layers::{Layer, LayerBuilder};
use crate::regularizer::Regularizers;

struct Dense {
    units: usize,
    activation: Option<Activation>,
    use_bias: bool,
    kernel_initializer: Option<Init>,
    bias_initializer: Option<Init>,
    activity_regularizer: Option<Regularizers>,
    bias_regularizer: Option<Regularizers>,
}

impl Dense {
    fn new(units: usize) -> Self {
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

    builder_field!(activation, Option<Activation>);
    builder_field!(use_bias, bool);
    builder_field!(kernel_initializer, Option<Init>);
    builder_field!(bias_initializer, Option<Init>);
    builder_field!(activity_regularizer, Option<Regularizers>);
    builder_field!(bias_regularizer, Option<Regularizers>);
}

impl LayerBuilder<DenseLayer> for Dense {
    fn build(&self, input_shape: &Shape, vs: &VarBuilder) -> Result<DenseLayer> {
        let input_dim = input_shape.dims().last().unwrap();
        let kernel =
            vs.get_with_hints(&[*input_dim, self.units], "kernel", DEFAULT_KAIMING_UNIFORM)?;
        let bias = match self.bias_initializer {
            Some(init) => {
                Some(vs.get_with_hints(&[self.units], "bias", DEFAULT_KAIMING_UNIFORM)?)
            }
            None => None,
        };

        Ok(DenseLayer {
            units: self.units,
            activation: self.activation,
            kernel,
            bias,
        })
    }
}

fn dense(units: usize) -> Dense {
    Dense::new(units)
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
