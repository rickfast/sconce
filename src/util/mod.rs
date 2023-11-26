use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("layer access invalid")]
    LayerAccess,
    #[error("layer index out of bounds")]
    LayerIndex,
    #[error("layer not found")]
    LayerNotFound,
}

#[macro_export]
macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field<'a>(&'a mut self, $field: $field_type) -> &'a mut Self {
            self.$field = $field;
            self
        }
    };
}
