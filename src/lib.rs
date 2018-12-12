// Copyright 2018 Ehsan M. Kermani.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "tvm_frontend"]
#![allow(
    non_camel_case_types,
    unused_assignments,
    unused_variables,
    unused_unsafe
)]
#![feature(try_from, fn_traits, unboxed_closures)]

//! [WIP]
//!
//! The `tvm_rust` crate aims at supporting Rust as one of the frontend API in
//! [TVM](https://github.com/dmlc/ts) runtime.
//!

extern crate tvm_sys as ts;
#[macro_use]
extern crate lazy_static;
extern crate custom_error;
extern crate ndarray as rust_ndarray;
extern crate num_traits;

use std::ffi::{CStr, CString};
use std::str;

/// Macro to check the return call to TVM runtime shared library
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", $crate::get_last_error());
        }
    }};
}

pub(crate) fn get_last_error() -> &'static str {
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

pub mod bytearray;
pub mod context;
pub mod errors;
pub mod function;
mod internal_api;
pub mod module;
pub mod ndarray;
pub mod ty;
pub mod value;

pub use bytearray::TVMByteArray;
pub use context::*;
pub use errors::{Error, Result};
pub use function::Function;
pub use module::Module;
pub use ndarray::{empty, NDArray};
pub use ty::*;
pub use value::*;

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
