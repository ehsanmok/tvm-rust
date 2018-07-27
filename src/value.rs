use super::*;

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum ValueKind {
    Int,
    Float,
    Handle,
    Str,
    Type,
    Context,
    Unknown
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
    ($type:ty, $kind:expr, $field:ident, $as:ty) => {
        impl<'a> From<&'a $type> for TVMValue {
            fn from(arg: &$type) -> Self {
                let inner = tvm::TVMValue { $field: *arg as $as };
                Self::new($kind, inner)
            }
        }
    };

    ($type:ty, $kind:expr, v_int64) => {
        impl_prim_val!($type, $kind, v_int64, i64);
    };
//    ($type:ty, v_float64) => {
//        impl_prim_val!($type, v_float64, f64);
//    };
}

impl_prim_val!(i64, ValueKind::Int, v_int64);
impl_prim_val!(i32, ValueKind::Int, v_int64);
impl_prim_val!(i8, ValueKind::Int, v_int64);
impl_prim_val!(u64, ValueKind::Int, v_int64);
impl_prim_val!(u32, ValueKind::Int, v_int64);
impl_prim_val!(u8, ValueKind::Int, v_int64);
impl_prim_val!(bool, ValueKind::Int, v_int64);
// impl_prim_val!(tvm::f64, v_float64);
impl<'a> From<&'a tvm::f32> for TVMValue {
    fn from(arg: &tvm::f32) -> Self {
        let inner = tvm::TVMValue { v_float64:  ordered_float::OrderedFloat((*arg).into_inner() as f64) };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a tvm::f64> for TVMValue {
    fn from(arg: &tvm::f64) -> Self {
        let inner = tvm::TVMValue { v_float64:  *arg };
        Self::new(ValueKind::Float, inner)
    }
}

impl<'a> From<&'a str> for TVMValue {
    fn from(arg: &str) -> TVMValue {
        let inner = tvm::TVMValue { v_str: arg.as_ptr() as *const c_char };
        Self::new(ValueKind::Str, inner)
    }
}

impl<'a> From<&'a String> for TVMValue {
    fn from(arg: &String) -> TVMValue {
        let inner = tvm::TVMValue { v_str: arg.as_ptr() as *const c_char };
        Self::new(ValueKind::Str, inner)
    }
}

impl<T> From<*mut T> for TVMValue {
    fn from(arg: *mut T) -> Self {
        let inner = tvm::TVMValue { v_handle: arg as *mut c_void };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<T> From<*const T> for TVMValue {
    fn from(arg: *const T) -> Self {
        let inner = tvm::TVMValue { v_handle: arg as *mut T as *mut c_void };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a mut tvm::TVMArray> for TVMValue {
    fn from(arr: &'a mut tvm::TVMArray) -> Self {
        let inner = tvm::TVMValue { v_handle: arr as *mut _ as *mut c_void };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a tvm::TVMArray> for TVMValue {
    fn from(arr: &'a tvm::TVMArray) -> Self {
        let inner = tvm::TVMValue { v_handle: arr as *const _ as *mut tvm::TVMArray as *mut c_void };
        Self::new(ValueKind::Handle, inner)
    }
}

impl<'a> From<&'a TVMContext> for TVMValue {
    fn from(ctx: &TVMContext) -> Self {
        let inner = tvm::TVMValue {
                    v_ctx: tvm::TVMContext {
                    device_type: ctx.device_type.into(),
                    device_id: ctx.device_id
                }};
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
        unimplemented!()
    }
}

impl Eq for TVMValue {}

impl Debug for TVMValue {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        unsafe { write!(f, "TVMValue: [v_int64: {:?}], [v_float64: {:?}], [v_handle: {:?}],\
                        [v_str: {:?}]", self.inner.v_int64, self.inner.v_float64,
                        self.inner.v_handle, self.inner.v_str) }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TVMArgValue {
    pub value: TVMValue,
    pub type_code: TypeCode,
}

impl TVMArgValue {
    pub fn new(value: TVMValue, type_code: TypeCode) -> Self {
        TVMArgValue { value, type_code }
    }
}

impl<'a> From<&'a TVMArgValue> for TVMValue {
    fn from(arg: &TVMArgValue) -> Self {
        arg.clone().value
    }
}

impl<'a> From<&'a TVMArgValue> for TypeCode {
    fn from(arg: &TVMArgValue) -> Self {
        arg.clone().type_code
    }
}

pub type TVMRetValue = TVMArgValue;

macro_rules! impl_prim_ret {
    ($type:ty, $type_code:expr) => {
        impl From<$type> for TVMRetValue {
            fn from(ret: $type) -> Self {
                TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&$type_code))
            }
        }
    }
}

impl_prim_ret!(i32, 0);
impl_prim_ret!(u32, 1);
impl_prim_ret!(tvm::f32, 2);
// impl_prim_ret!(f64, 2);
impl_prim_ret!(String, 11);