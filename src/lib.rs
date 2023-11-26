mod error;
mod layers;
mod regularizer;
mod util;

use crate::layers::Layer;
use crate::util::ModelError;
use crate::util::ModelError::{LayerAccess, LayerIndex, LayerNotFound};
use candle_core::{DType, Shape, Tensor};
use derive_builder::Builder;
use std::collections::HashMap;
use thiserror::Error;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

trait Model {
    fn call(
        &self,
        inputs: HashMap<String, Tensor>,
        parameters: HashMap<String, String>,
    ) -> HashMap<String, Tensor>;
    fn layers(&self) -> &Vec<Box<dyn Layer>>;
    fn get_layer(
        &self,
        name: Option<&str>,
        index: Option<usize>,
    ) -> Result<&Box<dyn Layer>, ModelError> {
        if (name.is_some() && index.is_some()) || (name.is_none() && index.is_none()) {
            return Err(LayerAccess);
        }

        let result = if name.is_some() {
            self.layers()
                .iter()
                .find(|layer| layer.name().eq(name.unwrap()))
        } else {
            self.layers().get(index.unwrap())
        };

        Ok(result.unwrap())
    }
}

pub struct Sequential<'a> {
    layers: &'a Vec<Box<dyn Layer>>,
    trainable: bool,
    functional: bool,
    name: String,
}

impl Model for Sequential<'_> {
    fn call(
        &self,
        inputs: HashMap<String, Tensor>,
        parameters: HashMap<String, String>,
    ) -> HashMap<String, Tensor> {
        todo!()
    }

    fn layers(&self) -> &Vec<Box<dyn Layer>> {
        return self.layers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
