use candle_core::Tensor;

pub type LossFn = fn(&Tensor, &Tensor) -> crate::error::Result<Tensor>;
