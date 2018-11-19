// Copyright 2018 Ehsan M. Kermani.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "tvm_rust"]
#![doc(html_root_url = "https://docs.rs/ts-rust/0.0.2/")]
#![allow(
    non_camel_case_types,
    dead_code,
    unused_assignments,
    unused_variables,
    unused_unsafe
)]
#![feature(try_from, fn_traits, unboxed_closures, box_syntax)]

//! [WIP]
//!
//! The `tvm_rust` crate aims at supporting Rust as one of the frontend API in
//! [TVM](https://github.com/dmlc/ts) runtime.
//!

pub extern crate tvm_sys as ts;
#[macro_use]
extern crate lazy_static;
extern crate ndarray as rust_ndarray;
extern crate num_traits;

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt::{self, Display, Formatter};
use std::result;
use std::str;

/// Macro to check the return call to TVM runtime shared library
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", ::TVMError::get_last());
        }
    }};
}

/// TVM error type
#[derive(Debug)]
pub struct TVMError {
    msg: &'static str,
}

impl TVMError {
    pub fn new(msg: &'static str) -> TVMError {
        TVMError { msg }
    }
    /// Get the last error message from TVM
    pub fn get_last() -> &'static str {
        unsafe {
            match CStr::from_ptr(ts::TVMGetLastError()).to_str() {
                Ok(s) => s,
                Err(_) => "Invalid UTF-8 message",
            }
        }
    }

    pub fn set_last(msg: &'static str) {
        let c_string = CString::new(msg).unwrap();
        unsafe {
            ts::TVMAPISetLastError(c_string.as_ptr());
        }
    }
}

impl Display for TVMError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", TVMError::get_last())
    }
}

impl Error for TVMError {
    fn description(&self) -> &'static str {
        TVMError::get_last()
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

pub mod context;
pub mod function;
mod internal_api;
pub mod module;
pub mod ndarray;
pub mod ty;
pub mod value;

pub use context::*;
// pub use function::{Function, BackendPackedCFunc};
pub use function::Function;
pub use module::Module;
pub use ndarray::{empty, NDArray};
pub use std::os::raw::{c_int, c_void};
pub use ty::*;
pub use value::*;

/// TVM result type
pub type TVMResult<T> = result::Result<T, TVMError>;

pub fn version() -> &'static str {
    str::from_utf8(ts::TVM_VERSION).unwrap()
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
        let msg: &'static str = "invalid error message";
        TVMError::set_last(msg);
        assert_eq!(TVMError::get_last().trim(), msg);
    }
}
