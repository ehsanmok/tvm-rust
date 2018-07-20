// Copyright 2018 Ehsan M. Kermani.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![crate_name = "tvm_rust"]
#![doc(html_root_url = "https://docs.rs/tvm-rust/0.0.2/")]
#![allow(
    non_camel_case_types,
    unused_imports,
    dead_code,
    unused_variables,
    unused_unsafe
)]
#![feature(try_from, fn_traits, unboxed_closures, box_syntax)]

//! [WIP]
//!
//! The `tvm_rust` crate aims at supporting Rust as one of the frontend API in
//! [TVM](https://github.com/dmlc/tvm) runtime.
//!

extern crate ndarray as rndarray;
extern crate ordered_float;
extern crate tvm_sys as tvm;

use std::convert::From;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::result;
use std::str;

/// Macro to check the return call to TVM runtime shared library
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

    pub fn set_last(msg: &'static str) {
        unsafe {
            tvm::TVMAPISetLastError(msg.as_ptr() as *const c_char);
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

pub mod function;
pub mod module;
pub mod ndarray;

pub use function::Function;
pub use module::Module;
pub use ndarray::NDArray;

type f32 = tvm::f32;
type f64 = tvm::f64;

/// TVM result type
pub type TVMResult<T> = result::Result<T, TVMError>;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TypeCode {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kHandle = 3,
    kNull = 4,
    kTVMType = 5,
    kTVMContext = 6,
    kArrayHandle = 7,
    kNodeHandle = 8,
    kModuleHandle = 9,
    kFuncHandle = 10,
    kStr = 11,
    kBytes = 12,
    kExtBegin = 15,
    kNNVMFirst = 16,
    kNNVMLast = 20,
    kExtReserveEnd = 64,
    kExtEnd = 128,
}

macro_rules! impl_prim_type {
    ($type:ty, $variant:ident) => {
        impl<'a> From<&'a $type> for TypeCode {
            fn from(arg: &$type) -> Self {
                TypeCode::$variant
            }
        }
    };
}

impl_prim_type!(i64, kDLInt);
impl_prim_type!(i32, kDLInt);
impl_prim_type!(i8, kDLInt);
impl_prim_type!(u64, kDLUInt);
impl_prim_type!(u32, kDLUInt);
impl_prim_type!(u8, kDLUInt);
impl_prim_type!(f64, kDLFloat);
impl_prim_type!(f32, kDLFloat);

trait TVMTypeCode: 'static {
    fn type_code() -> TypeCode;
}

#[derive(Clone)]
pub struct TVMValue {
    inner: tvm::TVMValue,
}

impl TVMValue {
    fn new(inner: tvm::TVMValue) -> Self {
        TVMValue { inner }
    }
}

macro_rules! impl_prim_val {
    ($type:ty, $field:ident, $as:ty) => {
        impl<'a> From<&'a $type> for TVMValue {
            fn from(arg: &$type) -> Self {
                TVMValue {
                    inner: tvm::TVMValue {
                        $field: *arg as $as,
                    },
                }
            }
        }
    };
    ($type:ty,v_int64) => {
        impl_prim_val!($type, v_int64, i64);
    };
    ($type:ty,v_float64) => {
        impl_prim_val!($type, v_float64, f64);
    };
}

impl_prim_val!(i64, v_int64);
impl_prim_val!(i32, v_int64);
impl_prim_val!(i8, v_int64);
impl_prim_val!(u64, v_int64);
impl_prim_val!(u32, v_int64);
impl_prim_val!(u8, v_int64);
impl_prim_val!(bool, v_int64);
impl_prim_val!(f64, v_float64);
// TODO: fix non-primitive cast for ordered_float
// impl_prim_val!(f32, v_float64);

impl<T> From<*mut T> for TVMValue {
    fn from(arg: *mut T) -> Self {
        TVMValue {
            inner: tvm::TVMValue {
                v_handle: arg as *mut c_void,
            },
        }
    }
}

impl<T> From<*const T> for TVMValue {
    fn from(arg: *const T) -> Self {
        TVMValue {
            inner: tvm::TVMValue {
                v_handle: arg as *mut T as *mut c_void,
            },
        }
    }
}

impl<'a> From<&'a mut tvm::TVMArray> for TVMValue {
    fn from(arr: &'a mut tvm::TVMArray) -> Self {
        TVMValue {
            inner: tvm::TVMValue {
                v_handle: arr as *mut _ as *mut c_void,
            },
        }
    }
}

impl<'a> From<&'a tvm::TVMArray> for TVMValue {
    fn from(arr: &'a tvm::TVMArray) -> Self {
        TVMValue {
            inner: tvm::TVMValue {
                v_handle: arr as *const _ as *mut tvm::TVMArray as *mut c_void,
            },
        }
    }
}

impl Hash for TVMValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            self.inner.v_int64.hash(state);
            self.inner.v_float64.hash(state);
            self.inner.v_handle.hash(state);
            self.inner.v_str.hash(state);
            self.inner.v_type.hash(state);
            self.inner.v_ctx.hash(state);
        }
    }
}

impl Debug for TVMValue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe { write!(f, "TVMValue") }
    }
}

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

impl From<tvm::DLDeviceType> for TVMDeviceType {
    fn from(device_type: tvm::DLDeviceType) -> Self {
        TVMDeviceType::new(device_type)
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

impl From<tvm::DLContext> for TVMContext {
    fn from(ctx: tvm::DLContext) -> Self {
        TVMContext {
            device_type: ctx.device_type.into(),
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

#[derive(Debug, Copy, Clone)]
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

impl From<tvm::DLDataType> for TVMType {
    fn from(dtype: tvm::DLDataType) -> Self {
        Self::new(dtype.code, dtype.bits, dtype.lanes)
    }
}

pub struct TVMArrayHandle {
    handle: tvm::TVMArrayHandle,
}

pub fn version() -> &'static str {
    str::from_utf8(tvm::TVM_VERSION).unwrap()
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
    fn print_version() {
        println!("TVM version: {}", version());
    }

    #[test]
    fn error() {
        let msg: &'static str = "Invalid";
        TVMError::set_last(msg);
        assert_eq!(TVMError::last().trim(), msg);
    }
}
