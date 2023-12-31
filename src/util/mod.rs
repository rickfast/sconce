#[macro_export]
macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field<'a>(&'a mut self, $field: $field_type) -> &'a mut Self {
            self.$field = $field;
            self
        }
    };
}

#[macro_export]
macro_rules! builder_option_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field<'a>(&'a mut self, $field: $field_type) -> &'a mut Self {
            self.$field = Some($field);
            self
        }
    };
}
