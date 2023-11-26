use candle_core::Tensor;
use candle_nn::loss::*;
use crate::error::Result;
use crate::loss::Loss;

mod sequential;

trait Model {
    fn compile(&self, loss: Loss);
}

struct X {

}

impl Model for X {
    fn compile(&self, loss: Loss) {
        todo!()
    }
}
