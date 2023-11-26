use crate::layers::core::dense::Dense;
use crate::layers::{LayerBuilder};
use candle_nn::init::ZERO;

struct Sequential {
    layers: Vec<Box<dyn LayerBuilder>>,
}

impl Sequential {
    fn new() -> Self {
        Sequential { layers: vec![] }
    }
    fn add_layer<L: LayerBuilder + Clone + 'static>(&mut self, layer: &mut L) -> &mut Self{
        self.layers.push(Box::new(layer.clone()));

        self
    }
}

fn do_it() {
    let _sequential = Sequential::new()
        .add_layer(Dense::new(2).kernel_initializer(Some(ZERO)));

}
