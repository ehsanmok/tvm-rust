use super::*;

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
impl_prim_type!(tvm::f64, kDLFloat);
impl_prim_type!(tvm::f32, kDLFloat);
impl_prim_type!(str, kStr);

pub(crate) trait TVMTypeCode {
    fn type_code() -> TypeCode;
}

#[derive(Debug, Copy, Clone)]
pub struct TVMType {
    pub(crate) inner: tvm::TVMType, // (type) code: u8, bits: u8, lanes: u16
}

impl TVMType {
    pub(crate) fn new(type_code: u8, bits: u8, lanes: u16) -> Self {
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
