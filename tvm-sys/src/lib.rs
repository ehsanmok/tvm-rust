#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code,
    improper_ctypes
)]

extern crate ordered_float;

mod helper {
    pub type f32_helper = f32;
    pub type f64_helper = f64;
}
pub type f32 = ordered_float::OrderedFloat<helper::f32_helper>;
pub type f64 = ordered_float::OrderedFloat<helper::f64_helper>;

include!("bindgen.rs");
