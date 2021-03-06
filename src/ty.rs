//! This module implements the required conversions from Rust types to TVM types.
//!
//! In TVM frontend only conversions from 32bits numeric types (i32, u32, f32)
//! and 64bits pointers are supported.
//!
//! # Example
//!
//! ```
//! let dtype = TVMType::from("float");
//! println!("dtype is: {}", dtype);
//! ```

use std::{
    ffi::{CStr, CString},
    fmt::{self, Display, Formatter},
    ops::{Deref, DerefMut},
};

use ts;

use function::Function;
use module::Module;
use ndarray::NDArray;
use TVMByteArray;
use TVMContext;
use TVMDeviceType;

/// TVM Type codes
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
}

impl Default for TypeCode {
    fn default() -> Self {
        TypeCode::kDLInt
    }
}

impl<'a> Into<TypeCode> for i32 {
    fn into(self) -> TypeCode {
        match self {
            0 => TypeCode::kDLInt,
            1 => TypeCode::kDLUInt,
            2 => TypeCode::kDLFloat,
            3 => TypeCode::kHandle,
            4 => TypeCode::kNull,
            5 => TypeCode::kTVMType,
            6 => TypeCode::kTVMContext,
            7 => TypeCode::kArrayHandle,
            8 => TypeCode::kNodeHandle,
            9 => TypeCode::kModuleHandle,
            10 => TypeCode::kFuncHandle,
            11 => TypeCode::kStr,
            12 => TypeCode::kBytes,
            _ => unreachable!(),
        }
    }
}

impl Display for TypeCode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TypeCode::kDLInt => "int",
                TypeCode::kDLUInt => "uint",
                TypeCode::kDLFloat => "float",
                TypeCode::kHandle => "handle",
                TypeCode::kNull => "null",
                TypeCode::kTVMType => "TVM type",
                TypeCode::kTVMContext => "TVM context",
                TypeCode::kArrayHandle => "Array handle",
                TypeCode::kNodeHandle => "Node handle",
                TypeCode::kModuleHandle => "Module handle",
                TypeCode::kFuncHandle => "Function handle",
                TypeCode::kStr => "string",
                TypeCode::kBytes => "bytes",
            }
        )
    }
}

macro_rules! impl_prim_type {
    ($type:ty, $variant:ident) => {
        impl<'a> From<&'a $type> for TypeCode {
            fn from(_arg: &$type) -> Self {
                TypeCode::$variant
            }
        }

        impl<'a> From<&'a mut $type> for TypeCode {
            fn from(_arg: &mut $type) -> Self {
                TypeCode::$variant
            }
        }
    };
}

impl_prim_type!(usize, kDLInt);
impl_prim_type!(i64, kDLInt);
impl_prim_type!(i32, kDLInt);
impl_prim_type!(i16, kDLInt);
impl_prim_type!(i8, kDLInt);
impl_prim_type!(TVMDeviceType, kDLInt);

impl_prim_type!(u64, kDLUInt);
impl_prim_type!(u32, kDLUInt);
impl_prim_type!(u16, kDLUInt);
impl_prim_type!(u8, kDLUInt);

impl_prim_type!(f64, kDLFloat);
impl_prim_type!(f32, kDLFloat);

impl_prim_type!(str, kStr);
impl_prim_type!(CStr, kStr);
impl_prim_type!(String, kStr);
impl_prim_type!(CString, kStr);

impl_prim_type!(TVMContext, kTVMContext);

impl_prim_type!(TVMType, kTVMType);

impl_prim_type!(Function, kFuncHandle);

impl_prim_type!(Module, kModuleHandle);

impl_prim_type!(NDArray, kArrayHandle);

impl_prim_type!([u8], kBytes);
impl_prim_type!(TVMByteArray, kBytes);

/// See the [module-level documentation](../ty/index.html) for more details.
///
/// Wrapper around underlying TVMType
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TVMType {
    pub inner: ts::TVMType, // fields are code: u8, bits: u8, lanes: u16
}

impl TVMType {
    pub(crate) fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
        TVMType {
            inner: ts::TVMType {
                code: type_code,
                bits: bits,
                lanes: lanes,
            },
        }
    }
}

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

impl Display for TVMType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self.inner {
            ts::TVMType {
                code: 0,
                bits: 32,
                lanes: 1,
            } => write!(f, "int"),
            ts::TVMType {
                code: 1,
                bits: 32,
                lanes: 1,
            } => write!(f, "uint"),
            ts::TVMType {
                code: 2,
                bits: 32,
                lanes: 1,
            } => write!(f, "float"),
            ts::TVMType {
                code: 4,
                bits: 64,
                lanes: 1,
            } => write!(f, "handle"),
            _ => write!(f, "unknown type"),
        }
    }
}

impl From<TVMType> for ts::DLDataType {
    fn from(dtype: TVMType) -> Self {
        dtype.inner
    }
}

impl From<ts::DLDataType> for TVMType {
    fn from(dtype: ts::DLDataType) -> Self {
        Self::new(dtype.code, dtype.bits, dtype.lanes)
    }
}

impl Deref for TVMType {
    type Target = ts::TVMType;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TVMType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, 'b> From<&'b TVMType> for &'a str {
    fn from(ty: &TVMType) -> Self {
        match **ty {
            ts::TVMType {
                code: 0,
                bits: 32,
                lanes: 1,
            } => "int",
            ts::TVMType {
                code: 1,
                bits: 32,
                lanes: 1,
            } => "uint",
            ts::TVMType {
                code: 2,
                bits: 32,
                lanes: 1,
            } => "float",
            ts::TVMType {
                code: 4,
                bits: 64,
                lanes: 1,
            } => "handle",
            _ => panic!("undefined type"),
        }
    }
}
