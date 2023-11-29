use thiserror::Error;

pub type Result<T> = candle_core::Result<T>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("layer access invalid")]
    LayerAccess,
    #[error("layer index out of bounds")]
    LayerIndex,
    #[error("layer not found")]
    LayerNotFound,
    #[error("model architecture must contain at least one layer")]
    EmptyModelArchitecture,
    #[error("model architecture does not include input shape")]
    InputShapeNotFound
}

impl From<Error> for candle_core::Error {
    fn from(value: Error) -> Self {
        candle_core::Error::Msg(value.to_string())
    }
}
