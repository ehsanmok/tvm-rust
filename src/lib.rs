//! [TVM](https://github.com/dmlc/tvm) is a compiler stack for deep learning systems.
//!
//! This crate provides idiomatic Rust API for TVM runtime frontend.
//!
//! One particular usage is that given an optimized deep learning model,
//! compiled with TVM, one can load the model artifacts which includes a shared library
//! `lib.so`, `graph.json` and byte-array `param.params`
//! in Rust to create a runtime, run the model for some inputs and get the
//! desired predictions all in Rust.
//!
//! Checkout the `examples` repository for more details.

#![crate_name = "tvm_frontend"]
#![allow(
    non_camel_case_types,
    unused_assignments,
    unused_variables,
    unused_unsafe
)]
#![feature(try_from, fn_traits, unboxed_closures)]

extern crate tvm_sys as ts;
#[macro_use]
extern crate lazy_static;
extern crate custom_error;
extern crate ndarray as rust_ndarray;
extern crate num_traits;

use std::ffi::{CStr, CString};
use std::str;

// Macro to check the return call to TVM runtime shared library
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", $crate::get_last_error());
        }
    }};
}

/// Gets the last error message
pub fn get_last_error() -> &'static str {
    unsafe {
        match CStr::from_ptr(ts::TVMGetLastError()).to_str() {
            Ok(s) => s,
            Err(_) => "Invalid UTF-8 message",
        }
    }
}

pub(crate) fn set_last_error(err: &Error) {
    let c_string = CString::new(err.to_string()).unwrap();
    unsafe {
        ts::TVMAPISetLastError(c_string.as_ptr());
    }
}

#[macro_use]
pub mod function;
pub mod bytearray;
pub mod context;
pub mod errors;
mod internal_api;
pub mod module;
pub mod ndarray;
pub mod ty;
pub mod value;

pub use bytearray::TVMByteArray;
pub use context::TVMContext;
pub use context::TVMDeviceType;
pub use errors::Error;
pub use errors::Result;
pub use function::Function;
pub use module::Module;
pub use ndarray::{empty, NDArray};
pub use ty::TVMType;
pub use value::TVMArgValue;
pub use value::TVMRetValue;

/// Outputs the current TVM version
pub fn version() -> &'static str {
    match str::from_utf8(ts::TVM_VERSION) {
        Ok(s) => s,
        Err(_) => "Invalid UTF-8 string",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_version() {
        println!("TVM version: {}", version());
    }

    #[test]
    fn set_error() {
        let err = Error::EmptyArray;
        set_last_error(&err);
        assert_eq!(get_last_error().trim(), err.to_string());
    }
}
