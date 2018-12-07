use std::ffi::{CStr, CString};
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_char, c_void};

use ts;

use Function;
use Module;
use NDArray;
use TVMByteArray;
use TVMContext;
use TVMDeviceType;
use TVMType;
use TypeCode;

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

// TODO: cleanup
impl<'a> From<&'a [u8]> for TVMValue {
    fn from(arg: &[u8]) -> TVMValue {
        let len = arg.len();
        let arg = CString::new(arg).unwrap();
        let arr = ts::TVMByteArray {
            data: arg.as_ptr() as *mut c_char,
            size: len,
        };
        let inner = ts::TVMValue {
            v_handle: &arr as *const _ as *mut c_void,
        };
        mem::forget(arr);
        mem::forget(arg);
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut [u8]> for TVMValue {
    fn from(arg: &mut [u8]) -> TVMValue {
        let len = arg.len();
        let arg = CString::new(arg).unwrap();
        let arr = ts::TVMByteArray {
            data: arg.as_ptr() as *mut c_char,
            size: len as usize,
        };
        let inner = ts::TVMValue {
            v_handle: &arg as *const _ as *mut c_void,
        };
        mem::forget(arg);
        Self::new(ValueKind::Handle, inner)
    }
}

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

impl<'a> From<&'a mut String> for TVMValue {
    fn from(arg: &mut String) -> TVMValue {
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
        let arg = arg.clone();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a mut CString> for TVMValue {
    fn from(arg: &mut CString) -> TVMValue {
        let arg = arg.clone();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a CStr> for TVMValue {
    fn from(arg: &CStr) -> TVMValue {
        let arg = arg.clone();
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        mem::forget(arg);
        Self::new(ValueKind::Str, inner)
    }
}

impl<T> From<*const T> for TVMValue {
    fn from(arg: *const T) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *mut T as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<T> From<*mut T> for TVMValue {
    fn from(arg: *mut T) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a Module> for TVMValue {
    fn from(arg: &Module) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg.handle as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut Module> for TVMValue {
    fn from(arg: &mut Module) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg.handle as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a Function> for TVMValue {
    fn from(arg: &Function) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *const _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut Function> for TVMValue {
    fn from(arg: &mut Function) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a NDArray> for TVMValue {
    fn from(arr: &NDArray) -> Self {
        let inner = ts::TVMValue {
            v_handle: arr.handle as *const _ as *mut ts::TVMArray as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut NDArray> for TVMValue {
    fn from(arr: &mut NDArray) -> Self {
        let inner = ts::TVMValue {
            v_handle: arr.handle as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a TVMType> for TVMValue {
    fn from(ty: &TVMType) -> Self {
        let inner = ts::TVMValue { v_type: ty.inner };
        Self::new(ValueKind::Type, inner)
    }
}

impl<'a> From<&'a mut TVMType> for TVMValue {
    fn from(ty: &mut TVMType) -> Self {
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

impl<'a> From<&'a mut TVMContext> for TVMValue {
    fn from(ctx: &mut TVMContext) -> Self {
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

impl<'a> From<&'a mut TVMDeviceType> for TVMValue {
    fn from(dev: &mut TVMDeviceType) -> Self {
        let inner = ts::TVMValue {
            v_int64: dev.0 as i64,
        };
        Self::new(ValueKind::Int, inner)
    }
}

impl<'a> From<&'a TVMByteArray> for TVMValue {
    fn from(barr: &TVMByteArray) -> Self {
        let barr = barr.clone();
        let inner = ts::TVMValue {
            v_handle: &barr.inner as *const ts::TVMByteArray as *mut c_void,
        };
        mem::forget(barr);
        Self::new(ValueKind::Bytes, inner)
    }
}

impl<'a> From<&'a mut TVMByteArray> for TVMValue {
    fn from(barr: &mut TVMByteArray) -> Self {
        let barr = barr.clone();
        let inner = ts::TVMValue {
            v_handle: &barr.inner as *const ts::TVMByteArray as *mut c_void,
        };
        mem::forget(barr);
        Self::new(ValueKind::Bytes, inner)
    }
}

impl PartialEq for TVMValue {
    fn eq(&self, other: &TVMValue) -> bool {
        if self.kind != other.kind {
            return false;
        }
        match &self {
            TVMValue { kind, inner } if kind == &ValueKind::Int => unsafe {
                inner.v_int64 == other.inner.v_int64
            },
            TVMValue { kind, inner } if kind == &ValueKind::Float => unsafe {
                inner.v_float64 == other.inner.v_float64
            },
            TVMValue { kind, inner } if kind == &ValueKind::Handle => unsafe {
                inner.v_handle == other.inner.v_handle
            },
            TVMValue { kind, inner } if kind == &ValueKind::Str => unsafe {
                inner.v_str == other.inner.v_str
            },
            TVMValue { kind, inner } if kind == &ValueKind::Type => unsafe {
                inner.v_type == other.inner.v_type
            },
            TVMValue { kind, inner } if kind == &ValueKind::Context => unsafe {
                inner.v_ctx == other.inner.v_ctx
            },
            _ => panic!("Undefined TVMValue comparision"),
        }
    }
}

impl Eq for TVMValue {}

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

#[derive(Debug, Clone, PartialEq, Eq)]
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

    pub fn to_bytearray(&self) -> Box<[u8]> {
        assert_eq!(
            self.type_code,
            TypeCode::kBytes,
            "Requires byte array, but found {:?}",
            self.type_code
        );
        unsafe {
            let barr_ptr =
                mem::transmute::<*mut c_void, *mut ts::TVMByteArray>(self.value.inner.v_handle);
            let barr = CStr::from_ptr((*barr_ptr).data).to_bytes();
            barr.to_vec().into_boxed_slice()
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
        let sptr: *const c_char = unsafe { self.value.inner.v_str };
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
}

impl<'b, 'a: 'b, T: 'b + ?Sized> From<&'b T> for TVMArgValue<'a>
where
    TVMValue: From<&'b T>,
    TypeCode: From<&'b T>,
{
    fn from(arg: &'b T) -> Self {
        TVMArgValue::new(TVMValue::from(arg), TypeCode::from(arg))
    }
}

pub type TVMRetValue<'a> = TVMArgValue<'a>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric() {
        let a = 42i8;
        let tvm_a = TVMArgValue::from(&a);
        assert_eq!(tvm_a.to_int() as i8, a);
        let a = 42i16;
        let tvm_a = TVMArgValue::from(&a);
        assert_eq!(tvm_a.to_int() as i16, a);
        let a = 42i32;
        let tvm_a = TVMArgValue::from(&a);
        assert_eq!(tvm_a.to_int() as i32, a);
        let a = 42i64;
        let tvm_a = TVMArgValue::from(&a);
        assert_eq!(tvm_a.to_int(), a);
        let b = 42f32;
        let tvm_b = TVMArgValue::from(&b);
        assert_eq!(tvm_b.to_float() as f32, b);
        let b = 42f64;
        let tvm_b = TVMArgValue::from(&b);
        assert_eq!(tvm_b.to_float(), b);
    }

    #[test]
    fn bytearray() {
        let v = CString::new(b"hello".to_vec()).unwrap();
        let v = v.into_bytes();
        let tvm = TVMArgValue::from(&v[..]);
        assert_eq!(tvm.to_bytearray(), v.into_boxed_slice());
        let w = vec![1u8, 2, 3, 4, 5];
        let tvm = TVMArgValue::from(&w[..]);
        assert_eq!(tvm.to_bytearray(), w.into_boxed_slice());
    }

    #[test]
    fn string() {
        let s = "hello";
        let tvm_arg = TVMArgValue::from(s);
        assert_eq!(tvm_arg.to_string(), s.to_string());
        let s = "hello".to_string();
        let tvm_arg = TVMArgValue::from(&s);
        assert_eq!(tvm_arg.to_string(), s);
    }

    #[test]
    fn ty() {
        let t = TVMType::from("int");
        let tvm = TVMArgValue::from(&t);
        assert_eq!(tvm.to_type(), t);
    }

    #[test]
    fn ctx() {
        let c = TVMContext::from("gpu");
        let tvm = TVMArgValue::from(&c);
        assert_eq!(tvm.to_ctx(), c);
    }
}
