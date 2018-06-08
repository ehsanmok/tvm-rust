extern crate tvm_sys as tvm;

use std::marker::PhantomData;
use std::mem;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use TVMArray;
use TVMContext;
use TVMError;
use TVMResult;
use TVMType;
use TypeCode;
use TVMTypeCode;

pub struct NDArray {
    handle: tvm::TVMArrayHandle,
    pub is_view: bool,
    pub dtype: TVMType,
}

impl TVMTypeCode for NDArray {
    fn type_code() -> TypeCode {
        TypeCode { sys: tvm::TVMTypeCode::kArrayHandle }
    }
}

impl NDArray {
    fn new(handle: tvm::TVMArrayHandle, is_view: bool, dtype: TVMType) -> Self {
        NDArray {
            handle: handle,
            is_view: is_view,
            dtype: dtype,
        }
    }

    pub fn empty(shape: &mut Vec<i32>, ctx: TVMContext, dtype: TVMType) -> TVMResult<Self> {
        let mut handle = ptr::null_mut() as tvm::TVMArrayHandle;
        let out = &mut handle as *mut tvm::TVMArrayHandle;
        check_call!(tvm::TVMArrayAlloc(
            shape.as_ptr() as *const i64,
            1 as c_int, // ?
            dtype.inner.code as c_int,
            dtype.inner.bits as c_int,
            dtype.inner.lanes as c_int,
            ctx.device_type.inner as c_int,
            ctx.device_id as c_int,
            out,
        ));
        Ok(Self::new(unsafe { *out }, false, dtype))
    }
}

impl Drop for NDArray {
    fn drop(&mut self) {
        check_call!(tvm::TVMArrayFree(self.handle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let shape = &mut vec![1, 2];
        let dtype = TVMType::from("float");
        let ctx = TVMContext::cpu(0);
        let e = NDArray::empty(shape, ctx, dtype);
        assert!(e.is_ok());
    }
}
