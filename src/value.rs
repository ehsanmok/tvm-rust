use std::ffi::OsString;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};

use super::*;

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum ValueKind {
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
    pub kind: ValueKind,
    pub(crate) inner: tvm::TVMValue,
}

impl TVMValue {
    pub(crate) fn new(kind: ValueKind, inner: tvm::TVMValue) -> Self {
        TVMValue { kind, inner }
    }
}

macro_rules! impl_prim_val {
    ($type:ty, $kind:expr, $field:ident) => {
        impl<'a> From<&'a $type> for TVMValue {
            fn from(arg: &$type) -> Self {
                let inner = tvm::TVMValue {
                    $field: *arg as i64,
                };
                Self::new($kind, inner)
            }
        }
    };
}

macro_rules! impl_prim_val_mut {
    ($type:ty, $kind:expr, $field:ident) => {
        impl<'a> From<&'a mut $type> for TVMValue {
            fn from(arg: &mut $type) -> Self {
                let inner = tvm::TVMValue {
                    $field: *arg as i64,
                };
                Self::new($kind, inner)
            }
        }
    };
}

impl_prim_val!(i64, ValueKind::Int, v_int64);
impl_prim_val!(i32, ValueKind::Int, v_int64);
impl_prim_val!(i8, ValueKind::Int, v_int64);
impl_prim_val!(u64, ValueKind::Int, v_int64);
impl_prim_val!(u32, ValueKind::Int, v_int64);
impl_prim_val!(u8, ValueKind::Int, v_int64);
impl_prim_val!(bool, ValueKind::Int, v_int64);
impl_prim_val!(tvm::DLDeviceType, ValueKind::Int, v_int64);

impl_prim_val_mut!(i64, ValueKind::Int, v_int64);
impl_prim_val_mut!(i32, ValueKind::Int, v_int64);
impl_prim_val_mut!(i8, ValueKind::Int, v_int64);
impl_prim_val_mut!(u64, ValueKind::Int, v_int64);
impl_prim_val_mut!(u32, ValueKind::Int, v_int64);
impl_prim_val_mut!(u8, ValueKind::Int, v_int64);
impl_prim_val_mut!(bool, ValueKind::Int, v_int64);
impl_prim_val_mut!(tvm::DLDeviceType, ValueKind::Int, v_int64);

impl<'a> From<&'a tvm::f32> for TVMValue {
    fn from(arg: &tvm::f32) -> Self {
        let inner = tvm::TVMValue {
            v_float64: ordered_float::OrderedFloat((*arg).into_inner() as f64),
        };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a mut tvm::f32> for TVMValue {
    fn from(arg: &mut tvm::f32) -> Self {
        let inner = tvm::TVMValue {
            v_float64: ordered_float::OrderedFloat((*arg).into_inner() as f64),
        };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a tvm::f64> for TVMValue {
    fn from(arg: &tvm::f64) -> Self {
        let inner = tvm::TVMValue { v_float64: *arg };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a mut tvm::f64> for TVMValue {
    fn from(arg: &mut tvm::f64) -> Self {
        let inner = tvm::TVMValue { v_float64: *arg };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a str> for TVMValue {
    fn from(arg: &str) -> TVMValue {
        let inner = tvm::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a String> for TVMValue {
    fn from(arg: &String) -> TVMValue {
        let inner = tvm::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a mut String> for TVMValue {
    fn from(arg: &mut String) -> TVMValue {
        let inner = tvm::TVMValue {
            v_str: arg.as_ptr() as *const c_char,
        };
        Self::new(ValueKind::Str, inner)
    }
}

impl<T> From<*mut T> for TVMValue {
    fn from(arg: *mut T) -> Self {
        let inner = tvm::TVMValue {
            v_handle: arg as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<T> From<*const T> for TVMValue {
    fn from(arg: *const T) -> Self {
        let inner = tvm::TVMValue {
            v_handle: arg as *mut T as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut Module> for TVMValue {
    fn from(arg: &mut Module) -> Self {
        let inner = tvm::TVMValue {
            v_handle: arg as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut Function> for TVMValue {
    fn from(arg: &mut Function) -> Self {
        let inner = tvm::TVMValue {
            v_handle: arg as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut NDArray> for TVMValue {
    fn from(arr: &mut NDArray) -> Self {
        let inner = tvm::TVMValue {
                v_handle: arr.handle as *mut _ as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a NDArray> for TVMValue {
    fn from(arr: &NDArray) -> Self {
        let inner = tvm::TVMValue {
                v_handle: arr.handle as *const _ as *mut tvm::TVMArray as *mut c_void,
        };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a TVMContext> for TVMValue {
    fn from(ctx: &TVMContext) -> Self {
        let inner = tvm::TVMValue {
            v_ctx: tvm::TVMContext {
                device_type: ctx.device_type.into(),
                device_id: ctx.device_id,
            },
        };
        Self::new(ValueKind::Context, inner)
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
            _ => panic!("Undefined value comparision"),
        }
    }
}

impl Eq for TVMValue {}

impl Default for TVMValue {
    fn default() -> Self {
        TVMValue::new(ValueKind::Int, tvm::TVMValue { v_int64: 0 })
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
    type Target = tvm::TVMValue;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for TVMValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TVMArgValue<'a> {
    pub value: TVMValue,
    pub type_code: TypeCode,
    _marker: PhantomData<&'a ()>,
}

impl<'a> TVMArgValue<'a> {
    pub fn new(value: TVMValue, type_code: TypeCode) -> Self {
        TVMArgValue {
            value: value,
            type_code: type_code,
            _marker: PhantomData,
        }
    }
}

impl<'a, 'b> From<&'b TVMArgValue<'a>> for TVMValue {
    fn from(arg: &TVMArgValue) -> Self {
        arg.clone().value
    }
}

impl<'a, 'b> From<&'b TVMArgValue<'a>> for TypeCode {
    fn from(arg: &TVMArgValue) -> Self {
        arg.clone().type_code
    }
}

pub type TVMRetValue<'a> = TVMArgValue<'a>;
