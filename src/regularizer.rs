use crate::error::Result;

use candle_core::{Module, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularizer {
    L1L2Default(),
    L1L2(f32, f32),
    L1(f32),
    L2(f32),
}

impl Module for Regularizer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::L1L2Default() => Self::L1L2(0.01, 0.01).forward(xs),
            &Self::L1L2(l1, l2) => {
                let mut regularization = Tensor::zeros(&[1], xs.dtype(), xs.device())?;
                let l1_t = Tensor::try_from(l1)?;
                let l2_t = Tensor::try_from(l2)?;

                if l1 != 0.0 {
                    regularization = regularization.add(&l1_t.mul(&xs.abs()?.sum_all()?)?)?;
                }

                if l2 != 0.0 {
                    regularization = regularization.add(&l2_t.mul(&xs.sqr()?.sum_all()?)?)?;
                }

                Ok(regularization)
            }
            &Self::L1(l1) => {
                let l1_t = Tensor::try_from(l1).unwrap();
                Ok(l1_t.mul(&xs.abs()?.sum_all()?)?)
            }
            &Self::L2(l2) => {
                let l2_t = Tensor::try_from(l2).unwrap();
                Ok(l2_t.mul(&xs.sqr()?.sum_all()?)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regularizer::Regularizer::L1;
    use candle_core::Device;

    #[test]
    fn test_l1() -> Result<()> {
        let x = Tensor::rand(0.0_f32, 1.0_f32, vec![4, 4], &Device::Cpu)?;
        let y = L1(0.1_f32).call(&x)?;
        let result = y.to_vec0::<f32>()?;
        let l1 = Tensor::try_from(0.1_f32)?;
        let expected = l1.mul(&x.abs()?.sum_all()?)?.to_vec0::<f32>()?;

        assert_eq!(result, expected);

        Ok(())
    }
}
