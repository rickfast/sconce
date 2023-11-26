use candle_core::Tensor;
use candle_nn::loss::*;
use crate::error::Result;

pub enum Loss {
    Nll,
    CrossEntropy,
    Mse,
    BinaryCrossEntropyWithLogit
}

impl LossFn for Loss {
    fn apply(&self, input: &Tensor, target: &Tensor) -> Result<Tensor> {
        match self {
            Loss::Nll => nll(input, target),
            Loss::CrossEntropy => cross_entropy(input, target),
            Loss::Mse => mse(input, target),
            Loss::BinaryCrossEntropyWithLogit => binary_cross_entropy_with_logit(input, target)
        }
    }
}

trait LossFn {
    fn apply(&self, input: &Tensor, target: &Tensor) -> Result<Tensor>;
}