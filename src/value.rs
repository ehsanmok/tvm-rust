//! This module implements [`TVMArgValue`] and [`TVMRetValue`] types and their conversions
//! to other support TVMValue. `TVMRetValue` is the owned version of `TVMArgValue`.
//!
//! # Examples
//!
//! ```
//! let a = 42i8;
//! let arg = TVMArgValue::from(&a);
//! assert_eq!(arg.to_int() as i8, a);
//! let ret = TVMRetValue::from(&a);
//! assert_eq!(ret.to_int() as i8, a);
//! ```

use std::{
    any::Any,
    ffi::{CStr, CString},
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    os::raw::{c_char, c_void},
};

use ts;

use ty::TypeCode;
use Function;
use Module;
use NDArray;
use TVMByteArray;
use TVMContext;
use TVMDeviceType;
use TVMType;

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Int,
    Float,
    Handle,
    Str,
    Bytes,
    Type,
    Context,
    Return,
}

/// Wrapper around underlying TVMValue.
#[derive(Clone)]
pub struct TVMValue {
    pub(crate) kind: ValueKind,
    pub(crate) inner: ts::TVMValue,
}

impl TVMValue {
    pub(crate) fn new(kind: ValueKind, inner: ts::TVMValue) -> Self {
        TVMValue { kind, inner }
    }

    pub fn to_int(&self) -> i64 {
        unsafe { self.inner.v_int64 }
    }
}

macro_rules! impl_prim_val {
    ($type:ty, $kind:expr, $field:ident, $cast:ty) => {
        impl<'a> From<&'a $type> for TVMValue {
            fn from(arg: &$type) -> Self {
                let inner = ts::TVMValue {
                    $field: *arg as $cast,
                };
                Self::new($kind, inner)
            }
        }

        impl<'a> From<&'a mut $type> for TVMValue {
            fn from(arg: &mut $type) -> Self {
                let inner = ts::TVMValue {
                    $field: *arg as $cast,
                };
                Self::new($kind, inner)
            }
        }
    };
}

impl_prim_val!(usize, ValueKind::Int, v_int64, i64);
impl_prim_val!(i64, ValueKind::Int, v_int64, i64);
impl_prim_val!(i32, ValueKind::Int, v_int64, i64);
impl_prim_val!(i16, ValueKind::Int, v_int64, i64);
impl_prim_val!(i8, ValueKind::Int, v_int64, i64);
impl_prim_val!(u64, ValueKind::Int, v_int64, i64);
impl_prim_val!(u32, ValueKind::Int, v_int64, i64);
impl_prim_val!(u16, ValueKind::Int, v_int64, i64);
impl_prim_val!(u8, ValueKind::Int, v_int64, i64);
impl_prim_val!(bool, ValueKind::Int, v_int64, i64);
impl_prim_val!(f64, ValueKind::Float, v_float64, f64);
impl_prim_val!(f32, ValueKind::Float, v_float64, f64);

impl<'a> From<&'a str> for TVMValue {
    fn from(arg: &str) -> TVMValue {
        let arg = CString::new(arg).unwrap();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a String> for TVMValue {
    fn from(arg: &String) -> TVMValue {
        let arg = CString::new(arg.as_bytes()).unwrap();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a CString> for TVMValue {
    fn from(arg: &CString) -> TVMValue {
        let arg = arg.to_owned();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a [u8]> for TVMValue {
    fn from(arg: &[u8]) -> TVMValue {
        let arg = arg.to_owned();
        let inner = ts::TVMValue {
            v_handle: &arg as *const _ as *mut c_void,
        };
        mem::forget(arg);
        Self::new(ValueKind::Handle, inner)
    }
}

macro_rules! impl_tvm_val_from_handle {
    ($($ty:ty),+) => {
        $(
            impl<'a> From<&'a $ty> for TVMValue {
                fn from(arg: &$ty) -> Self {
                    let inner = ts::TVMValue {
                        v_handle: arg.handle as *mut _ as *mut c_void,
                    };
                    Self::new(ValueKind::Handle, inner)
                }
            }
        )+
    }
}

impl_tvm_val_from_handle!(Module, Function, NDArray);

impl<'a> From<&'a TVMType> for TVMValue {
    fn from(ty: &TVMType) -> Self {
        let inner = ts::TVMValue { v_type: ty.inner };
        Self::new(ValueKind::Type, inner)
    }
}

impl<'a> From<&'a TVMContext> for TVMValue {
    fn from(ctx: &TVMContext) -> Self {
        let inner = ts::TVMValue {
            v_ctx: ctx.clone().into(),
        };
        Self::new(ValueKind::Context, inner)
    }
}

impl<'a> From<&'a TVMDeviceType> for TVMValue {
    fn from(dev: &TVMDeviceType) -> Self {
        let inner = ts::TVMValue {
            v_int64: dev.0 as i64,
        };
        Self::new(ValueKind::Int, inner)
    }
}

impl<'a> From<&'a TVMByteArray> for TVMValue {
    fn from(barr: &TVMByteArray) -> Self {
        let inner = ts::TVMValue {
            v_handle: &barr.inner as *const ts::TVMByteArray as *mut c_void,
        };
        Self::new(ValueKind::Bytes, inner)
    }
}

impl Debug for TVMValue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe {
            write!(
                f,
                "TVMValue: [v_int64: {:?}], [v_float64: {:?}], [v_handle: {:?}],\
                 [v_str: {:?}]",
                self.inner.v_int64, self.inner.v_float64, self.inner.v_handle, self.inner.v_str
            )
        }
    }
}

impl Deref for TVMValue {
    type Target = ts::TVMValue;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TVMValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// This type is needed for passing supported values as arguments to [`call_packed!`]
/// or [`function::Builder`]. Checkout the methods and from conversions.
///
/// ## Example
///
/// ```
/// let ctx = TVMContext::gpu(0);
/// let arg = TVMArgValue::from(&ctx);
/// assert_eq!(arg.to_ctx(), ctx);
/// ```
///
/// [`function::Builder`]:../function/struct.Builder.html
#[derive(Debug, Clone)]
pub struct TVMArgValue<'a> {
    pub value: TVMValue,
    pub type_code: TypeCode,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> TVMArgValue<'a> {
    pub fn new(value: TVMValue, type_code: TypeCode) -> Self {
        TVMArgValue {
            value: value,
            type_code: type_code,
            _lifetime: PhantomData,
        }
    }
}

/// Main way to create a TVMArgValue from suported Rust's values.
impl<'b, 'a: 'b, T: 'b + ?Sized> From<&'b T> for TVMArgValue<'a>
where
    TVMValue: From<&'b T>,
    TypeCode: From<&'b T>,
{
    fn from(arg: &'b T) -> Self {
        TVMArgValue::new(TVMValue::from(arg), TypeCode::from(arg))
    }
}

/// TVMRetValue is an owned TVMArgValue.
///
/// ## Example
///
/// ```
/// let ctx = TVMContext::gpu(0);
/// let arg = TVMRetValue::from(&ctx);
/// assert_eq!(arg.to_ctx(), ctx);
/// ```
#[derive(Debug)]
pub struct TVMRetValue {
    pub value: TVMValue,
    box_value: Box<Any>,
    pub type_code: TypeCode,
}

impl TVMRetValue {
    pub(crate) fn new(value: TVMValue, type_code: TypeCode) -> Self {
        Self {
            value,
            box_value: box (),
            type_code,
        }
    }
}

impl Clone for TVMRetValue {
    fn clone(&self) -> TVMRetValue {
        TVMRetValue {
            value: self.value.clone(),
            box_value: box (),
            type_code: self.type_code,
        }
    }
}

impl<'b, T: 'b + ?Sized> From<&'b T> for TVMRetValue
where
    TVMValue: From<&'b T>,
    TypeCode: From<&'b T>,
{
    fn from(arg: &'b T) -> Self {
        TVMRetValue::new(TVMValue::from(arg), TypeCode::from(arg))
    }
}

macro_rules! impl_to_methods {
    ($ty:ty) => {
        pub fn to_int(&self) -> i64 {
            if self.type_code != TypeCode::kDLInt && self.type_code != TypeCode::kNull {
                panic!("Requires i64 or NULL, but found {:?}", self.type_code);
            }

            unsafe { self.value.inner.v_int64 }
        }

        pub fn to_float(&self) -> f64 {
            assert_eq!(
                self.type_code,
                TypeCode::kDLFloat,
                "Requires f64, but found {:?}",
                self.type_code
            );
            unsafe { self.value.inner.v_float64 }
        }

        pub fn to_bytearray(&self) -> TVMByteArray {
            assert_eq!(
                self.type_code,
                TypeCode::kBytes,
                "Requires byte array, but found {:?}",
                self.type_code
            );
            unsafe {
                let barr_ptr =
                    mem::transmute::<*mut c_void, *mut ts::TVMByteArray>(self.value.inner.v_handle);
                TVMByteArray::new(*barr_ptr)
            }
        }

        pub fn to_module(&self) -> Module {
            assert_eq!(
                self.type_code,
                TypeCode::kModuleHandle,
                "Requires module handle, but found {:?}",
                self.type_code
            );
            let module_handle = unsafe { self.value.inner.v_handle };
            Module::new(module_handle, false, None)
        }

        pub fn to_string(&self) -> String {
            assert_eq!(
                self.type_code,
                TypeCode::kStr,
                "Requires string, but found {:?}",
                self.type_code
            );
            let ret_str = unsafe {
                match CStr::from_ptr(self.value.inner.v_str).to_str() {
                    Ok(s) => s,
                    Err(_) => "Invalid UTF-8 message",
                }
            };
            ret_str.to_string()
        }

        pub fn to_ndarray(&self) -> NDArray {
            assert_eq!(
                self.type_code,
                TypeCode::kArrayHandle,
                "Requires Array handle, but found {:?}",
                self.type_code
            );
            let handle = unsafe { self.value.inner.v_handle };
            let arr_handle = unsafe { mem::transmute::<*mut c_void, ts::TVMArrayHandle>(handle) };
            NDArray::new(arr_handle, true)
        }

        pub fn to_type(&self) -> TVMType {
            assert_eq!(
                self.type_code,
                TypeCode::kTVMType,
                "Requires TVMType, but found {:?}",
                self.type_code
            );
            let ty = unsafe { self.value.inner.v_type };
            TVMType::from(ty)
        }

        pub fn to_ctx(&self) -> TVMContext {
            assert_eq!(
                self.type_code,
                TypeCode::kTVMContext,
                "Requires TVMContext, but found {:?}",
                self.type_code
            );
            let ctx = unsafe { self.value.inner.v_ctx };
            TVMContext::from(ctx)
        }
    };

    (refnc $ty:ty) => {
        impl<'a> $ty {
            impl_to_methods!($ty);
        }
    };

    (owned $ty:ty) => {
        impl $ty {
            impl_to_methods!($ty);
        }
    }
}

impl_to_methods!(refnc TVMArgValue<'a>);
impl_to_methods!(owned TVMRetValue);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric() {
        macro_rules! arg_ret_tests {
            ($v:expr; ints $($ty:ty),+) => {{
                $(
                    let v = $v as $ty;
                    let a = TVMArgValue::from(&v);
                    assert_eq!(a.to_int() as $ty, v);
                    let b = TVMRetValue::from(&v);
                    assert_eq!(b.to_int() as $ty, v);
                )+
            }};
            ($v:expr; floats $($ty:ty),+) => {{
                $(
                    let v = $v as $ty;
                    let a = TVMArgValue::from(&v);
                    assert_eq!(a.to_float() as $ty, v);
                    let b = TVMRetValue::from(&v);
                    assert_eq!(b.to_float() as $ty, v);
                )+
            }};
        }

        arg_ret_tests!(42; ints i8, i16, i32, i64);
        arg_ret_tests!(42; floats f32, f64);
    }

    #[test]
    fn bytearray() {
        let v = CString::new(b"hello".to_vec()).unwrap();
        let v = v.into_bytes();
        let tvm = TVMRetValue::from(&v[..]);
        assert_eq!(
            tvm.to_bytearray().data(),
            v.iter().map(|e| *e as i8).collect::<Vec<i8>>()
        );
        let w = vec![1u8, 2, 3, 4, 5];
        let tvm = TVMRetValue::from(&w[..]);
        assert_eq!(
            tvm.to_bytearray().data(),
            w.iter().map(|e| *e as i8).collect::<Vec<i8>>()
        );
    }

    #[test]
    fn string() {
        let s = "hello";
        let tvm_arg = TVMRetValue::from(s);
        assert_eq!(tvm_arg.to_string(), s.to_string());
        let s = "hello".to_string();
        let tvm_arg = TVMRetValue::from(&s);
        assert_eq!(tvm_arg.to_string(), s);
    }

    #[test]
    fn ty() {
        let t = TVMType::from("int");
        let tvm = TVMRetValue::from(&t);
        assert_eq!(tvm.to_type(), t);
    }

    #[test]
    fn ctx() {
        let c = TVMContext::from("gpu");
        let tvm = TVMRetValue::from(&c);
        assert_eq!(tvm.to_ctx(), c);
    }
}
