// Copyright 2018 Ehsan M. Kermani.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "tvm_rust"]
#![doc(html_root_url = "https://docs.rs/tvm-rust/0.0.1/tvm-rust")]
#![allow(non_camel_case_types, unused_imports, unused_unsafe)]

//! [WIP]
//!
//! The `tvm_rust` crate aims at supporting Rust as one of the frontend API in
//! [TVM](https://github.com/dmlc/tvm) runtime.
//!

extern crate libc;
extern crate tvm_sys as tvm;

use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fmt::{Display, Formatter};

/// Macro to check the return call to TVM runtime C lib
#[macro_export]
macro_rules! check_call {
    ($e:expr) => {{
        if unsafe { $e } != 0 {
            panic!("{}", ::TVMError::last());
        }
    }};
}

/// TVM error type
#[derive(Debug)]
pub struct TVMError;

impl TVMError {
    /// Get the last error message from TVM
    pub fn last() -> String {
        unsafe {
            CStr::from_ptr(tvm::TVMGetLastError())
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Display for TVMError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", TVMError::last())
    }
}

impl Error for TVMError {
    fn description(&self) -> &str {
        unsafe { CStr::from_ptr(tvm::TVMGetLastError()).to_str().unwrap() }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

/// TVM result type
pub type TVMResult<T> = ::std::result::Result<T, TVMError>;

/// Type of devices supported by TVM. Default is cpu.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum DeviceType {
    Cpu = 1,
    Gpu = 2,
    Opencl = 4,
    Metal = 8,
    Vpi = 9,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
    }
}

/// TVM context. Default is cpu.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Context {
    /// Supported devices
    pub device_type: DeviceType,
    /// Device id
    pub device_id: i32,
}

// TODO: reexport devs
impl Context {
    pub fn new(device_type: DeviceType, device_id: i32) -> Self {
        Context {
            device_type: device_type,
            device_id: device_id,
        }
    }

    pub fn current_context(&self) -> &Self {
        self
    }

    pub fn cpu(device_id: i32) -> Self {
        Context {
            device_id: device_id,
            ..Default::default()
        }
    }

    pub fn gpu(device_id: i32) -> Self {
        Self::new(DeviceType::Gpu, device_id)
    }

    pub fn opencl(device_id: i32) -> Self {
        Self::new(DeviceType::Opencl, device_id)
    }

    pub fn vpi(device_id: i32) -> Self {
        Self::new(DeviceType::Vpi, device_id)
    }
}

impl Display for Context {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // TODO: display DeviceType
        write!(f, "{:?} {}", self.device_type, self.device_id)
    }
}
