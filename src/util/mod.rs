use thiserror::Error;

#[macro_export]
macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field<'a>(&'a mut self, $field: $field_type) -> &'a mut Self {
            self.$field = $field;
            self
        }
    };
}
