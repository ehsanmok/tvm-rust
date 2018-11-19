use std::ffi::CStr;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::os::raw::{c_char, c_void};

use ts;

use Function;
use Module;
use NDArray;
use TypeCode;

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum ValueKind {
    Int,
    Float,
    Handle,
    Str,
    Type,
    Context,
    Return,
    Unknown,
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
    };
}

macro_rules! impl_prim_val_mut {
    ($type:ty, $kind:expr, $field:ident, $cast:ty) => {
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

impl_prim_val!(i64, ValueKind::Int, v_int64, i64);
impl_prim_val!(i32, ValueKind::Int, v_int64, i64);
impl_prim_val!(i8, ValueKind::Int, v_int64, i64);
impl_prim_val!(u64, ValueKind::Int, v_int64, i64);
impl_prim_val!(u32, ValueKind::Int, v_int64, i64);
impl_prim_val!(u8, ValueKind::Int, v_int64, i64);
impl_prim_val!(bool, ValueKind::Int, v_int64, i64);

impl_prim_val!(f64, ValueKind::Float, v_float64, f64);
impl_prim_val!(f32, ValueKind::Float, v_float64, f64);

impl_prim_val_mut!(i64, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(i32, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(i8, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(u64, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(u32, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(u8, ValueKind::Int, v_int64, i64);
impl_prim_val_mut!(bool, ValueKind::Int, v_int64, i64);

impl_prim_val_mut!(f64, ValueKind::Float, v_float64, f64);
impl_prim_val_mut!(f32, ValueKind::Float, v_float64, f64);

impl<'a> From<&'a [u8]> for TVMValue {
    fn from(arg: &[u8]) -> TVMValue {
        let inner = ts::TVMValue {
            v_handle: arg.as_ptr() as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut [u8]> for TVMValue {
    fn from(arg: &mut [u8]) -> TVMValue {
        let inner = ts::TVMValue {
            v_handle: arg.as_mut_ptr() as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a str> for TVMValue {
    fn from(arg: &str) -> TVMValue {
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a String> for TVMValue {
    fn from(arg: &String) -> TVMValue {
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a mut String> for TVMValue {
    fn from(arg: &mut String) -> TVMValue {
        let inner = ts::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
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

impl<T> From<*const T> for TVMValue {
    fn from(arg: *const T) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *mut T as *mut c_void,
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

impl<'a> From<&'a mut Function> for TVMValue {
    fn from(arg: &mut Function) -> Self {
        let inner = ts::TVMValue {
            v_handle: arg as *mut _ as *mut c_void,
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

impl<'a> From<&'a NDArray> for TVMValue {
    fn from(arr: &NDArray) -> Self {
        let inner = ts::TVMValue {
            v_handle: arr.handle as *const _ as *mut ts::TVMArray as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
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

impl Default for TVMValue {
    fn default() -> Self {
        TVMValue::new(ValueKind::Int, ts::TVMValue { v_int64: 0 })
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
            panic!("Requires i64 or NULL, but given {}", self.type_code);
        }
        unsafe { self.value.inner.v_int64 }
    }

    pub fn to_float(&self) -> f64 {
        assert_eq!(
            self.type_code,
            TypeCode::kDLFloat,
            "Requires f64, but given {}",
            self.type_code
        );
        unsafe { self.value.inner.v_float64 }
    }

    pub fn to_bytes(&self) -> Box<[u8]> {
        assert_eq!(
            self.type_code,
            TypeCode::kBytes,
            "Requires byte array, but given {}",
            self.type_code
        );
        let arr = unsafe { mem::transmute::<*mut c_void, *mut i8>(self.value.inner.v_handle) };
        let barr: &[u8] = unsafe { CStr::from_ptr(arr).to_bytes() };
        barr.to_vec().into_boxed_slice()
    }

    pub fn to_module(&self) -> Module {
        assert_eq!(
            self.type_code,
            TypeCode::kModuleHandle,
            "Requires module handle, given {}",
            self.type_code
        );
        let module_handle = unsafe { self.value.inner.v_handle };
        Module::new(module_handle, false, None)
    }

    pub fn to_string(&self) -> String {
        assert_eq!(
            self.type_code,
            TypeCode::kStr,
            "Requires string, given {}",
            self.type_code
        );
        let sptr: *const c_char = unsafe { self.value.inner.v_str };
        let ret_str = unsafe {
            match CStr::from_ptr(sptr).to_str() {
                Ok(s) => s,
                Err(_) => "Invalid UTF-8 message",
            }
        };
        ret_str.to_owned()
    }

    pub fn to_ndarray(&self) -> NDArray {
        assert_eq!(
            self.type_code,
            TypeCode::kArrayHandle,
            "Requires Array handle, given {}",
            self.type_code
        );
        let handle = unsafe { self.value.inner.v_handle };
        let arr_handle = unsafe { mem::transmute::<*mut c_void, ts::TVMArrayHandle>(handle) };
        NDArray::new(arr_handle, true)
    }
}

impl<'b, 'a: 'b, T: 'b> From<&'b T> for TVMArgValue<'a>
where
    TVMValue: From<&'b T>,
    TypeCode: From<&'b T>,
{
    fn from(arg: &'b T) -> Self {
        TVMArgValue::new(TVMValue::from(arg), TypeCode::from(arg))
    }
}

pub type TVMRetValue<'a> = TVMArgValue<'a>;
