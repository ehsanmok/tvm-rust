// Copyright 2018 Ehsan M. Kermani.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "tvm_rust"]
#![doc(html_root_url = "https://docs.rs/tvm-rust/0.0.2/")]
#![allow(non_camel_case_types, unused_imports, dead_code, unused_variables, unused_unsafe)]

//! [WIP]
//!
//! The `tvm_rust` crate aims at supporting Rust as one of the frontend API in
//! [TVM](https://github.com/dmlc/tvm) runtime.
//!

extern crate libc;
extern crate tvm_sys as tvm;

use std::convert::From;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fmt::{Display, Formatter};
use std::os::raw::{c_int, c_void};
use std::ptr;

/// Macro to check the return call to TVM runtime shared library
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
    pub fn last() -> &'static str {
        unsafe {
            match CStr::from_ptr(tvm::TVMGetLastError()).to_str() {
                Ok(s) => s,
                Err(_) => "Invalid UTF-8 message",
            }
        }
    }
}

impl Display for TVMError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", TVMError::last())
    }
}

impl Error for TVMError {
    fn description(&self) -> &'static str {
        TVMError::last()
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

/// TVM result type
pub type TVMResult<T> = ::std::result::Result<T, TVMError>;

pub mod ndarray;

/// Type of devices supported by TVM. Default is cpu.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TVMDeviceType {
    inner: tvm::DLDeviceType,
}

impl TVMDeviceType {
    pub fn new(device_type: tvm::DLDeviceType) -> Self {
        TVMDeviceType { inner: device_type }
    }
}

impl Default for TVMDeviceType {
    fn default() -> Self {
        TVMDeviceType::new(tvm::DLDeviceType::kDLCPU)
    }
}

impl From<TVMDeviceType> for tvm::DLDeviceType {
    fn from(device_type: TVMDeviceType) -> Self {
        device_type.inner
    }
}

// TODO: include extTypes
impl<'a> From<&'a str> for TVMDeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => Self::new(tvm::DLDeviceType::kDLCPU),
            "llvm" => Self::new(tvm::DLDeviceType::kDLCPU),
            "stackvm" => Self::new(tvm::DLDeviceType::kDLCPU),
            "gpu" => Self::new(tvm::DLDeviceType::kDLGPU),
            "cuda" => Self::new(tvm::DLDeviceType::kDLGPU),
            "nvptx" => Self::new(tvm::DLDeviceType::kDLGPU),
            "cl" => Self::new(tvm::DLDeviceType::kDLOpenCL),
            "opencl" => Self::new(tvm::DLDeviceType::kDLOpenCL),
            "metal" => Self::new(tvm::DLDeviceType::kDLMetal),
            "vpi" => Self::new(tvm::DLDeviceType::kDLVPI),
            "rocm" => Self::new(tvm::DLDeviceType::kDLROCM),
            _ => panic!("{:?} not supported!", type_str),
        }
    }
}

/// TVM context. Default is cpu.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct TVMContext {
    /// Supported devices
    pub device_type: TVMDeviceType,
    /// Device id
    pub device_id: i32,
}

// TODO: reexport devs
impl TVMContext {
    pub fn new(device_type: TVMDeviceType, device_id: i32) -> Self {
        TVMContext {
            device_type: device_type,
            device_id: device_id,
        }
    }

    pub fn current_context(&self) -> &Self {
        self
    }

    pub fn cpu(device_id: i32) -> Self {
        TVMContext {
            device_id: device_id,
            ..Default::default()
        }
    }
    // TODO: refactor to macro impl
    pub fn gpu(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLGPU), device_id)
    }

    pub fn cpu_pinned(device_id: i32) -> Self {
        Self::new(
            TVMDeviceType::new(tvm::DLDeviceType::kDLCPUPinned),
            device_id,
        )
    }

    pub fn opencl(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLOpenCL), device_id)
    }

    pub fn metal(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLMetal), device_id)
    }

    pub fn vpi(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLVPI), device_id)
    }

    pub fn rocm(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLROCM), device_id)
    }
}

impl TVMContext {
    // TODO: impl with macro
    pub fn exist(&self) -> TVMResult<Self> {
        unimplemented!()
    }

    pub fn max_threads_per_block(&self) -> TVMResult<Self> {
        unimplemented!()
    }

    pub fn warp_size(&self) -> TVMResult<Self> {
        unimplemented!()
    }

    pub fn max_shared_memory_per_block(&self) -> TVMResult<Self> {
        unimplemented!()
    }

    pub fn compute_version(&self) -> TVMResult<Self> {
        unimplemented!()
    }
    // device_name
    // max_clock_rate
    // multi_processor_count
}

impl TVMContext {
    pub fn sync(&self) -> TVMResult<()> {
        check_call!(tvm::TVMSynchronize(
            self.device_type.inner as i32,
            self.device_id,
            ptr::null_mut()
        ));
        Ok(())
    }
}

impl From<TVMContext> for tvm::DLContext {
    fn from(ctx: TVMContext) -> Self {
        tvm::DLContext {
            device_type: tvm::DLDeviceType::from(ctx.device_type),
            device_id: ctx.device_id,
        }
    }
}

impl Display for TVMContext {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // TODO: display DeviceType
        write!(f, "{:?} {}", self.device_type, self.device_id)
    }
}

#[derive(Debug, Clone)]
pub struct TVMType {
    inner: tvm::TVMType, // (type) code: u8, bits: u8, lanes: u16
}

impl TVMType {
    pub fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
        TVMType {
            inner: tvm::TVMType {
                code: type_code,
                bits: bits,
                lanes: lanes,
            },
        }
    }
}

// only lanes = 1 for now
impl<'a> From<&'a str> for TVMType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "int" => TVMType::new(0, 32, 1),
            "uint" => TVMType::new(1, 32, 1),
            "float" => TVMType::new(2, 32, 1),
            "handle" => TVMType::new(4, 64, 1),
            _ => panic!("Unsupported type {:?}", type_str),
        }
    }
}

impl From<TVMType> for tvm::DLDataType {
    fn from(dtype: TVMType) -> Self {
        dtype.inner
    }
}

pub struct TVMArrayHandle {
    handle: tvm::TVMArrayHandle,
}

pub struct TVMArray {
    raw: tvm::TVMArray,
}

impl TVMArray {
    fn new(shape: &mut [i32], ctx: TVMContext, dtype: TVMType) -> Self {
        let raw = tvm::TVMArray {
            data: ptr::null_mut() as *mut c_void,
            ctx: tvm::DLContext::from(ctx),
            ndim: shape.len() as c_int,
            dtype: tvm::DLDataType::from(dtype),
            shape: shape.as_mut_ptr() as *mut i64,
            strides: ptr::null_mut() as *mut i64,
            byte_offset: 0u64,
        };
        TVMArray { raw }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = TVMContext::cpu(0);
        let default_ctx = TVMContext::new(TVMDeviceType::new(tvm::DLDeviceType::kDLCPU), 0);
        assert_eq!(ctx.current_context().clone(), default_ctx);
        assert_ne!(ctx, TVMContext::gpu(0));

        let str_ctx = TVMContext::new(TVMDeviceType::from("gpu"), 0);
        assert_eq!(str_ctx.current_context().clone(), str_ctx);
        assert_ne!(str_ctx, TVMContext::new(TVMDeviceType::from("cpu"), 0));
    }

    #[test]
    fn array() {
        let shape = &mut [1, 2];
        let ctx = TVMContext::cpu(0);
        let dtype = TVMType::from("float");
        let empty = TVMArray::new(shape, ctx, dtype);
        assert!(empty.raw.data.is_null());
        assert_eq!(empty.raw.ndim, shape.len() as i32);
    }
}
