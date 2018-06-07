extern crate tvm_sys as tvm;

use std::marker::PhantomData;
use std::mem;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

pub use TVMArray;
pub use TVMContext;
pub use TVMError;
pub use TVMResult;
pub use TVMType;

pub struct NDArray {
    handle: tvm::TVMArrayHandle,
    pub is_view: bool,
    pub dtype: TVMType,
}

impl NDArray {
    // TODO: fix
    pub fn empty(shape: &mut [i32], ctx: TVMContext, dtype: TVMType) -> TVMResult<Self> {
        let mut empty = TVMArray::new(shape, ctx.clone(), dtype.clone());
        let mut handle = &mut empty.raw as *mut tvm::TVMArray;
        // let out = &mut handle as *mut tvm::TVMArrayHandle;
        // let out: *mut tvm::TVMArrayHandle = unsafe { mem::transmute(&mut handle) };
        check_call!(tvm::TVMArrayAlloc(
            shape.as_ptr() as *const i64,
            shape.len() as c_int,
            dtype.inner.code as c_int,
            dtype.inner.bits as c_int,
            dtype.inner.lanes as c_int,
            ctx.device_type.inner as c_int,
            ctx.device_id,
            &mut handle as *mut tvm::TVMArrayHandle,
        ));
        Ok(Self::new(handle, false, dtype))
    }

    fn new(handle: tvm::TVMArrayHandle, is_view: bool, dtype: TVMType) -> Self {
        NDArray {
            handle: handle,
            is_view: is_view,
            dtype: dtype,
        }
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
        let shape = &mut [1, 2];
        let dtype = TVMType::from("float");
        let ctx = TVMContext::cpu(0);
        // error
        // let e = NDArray::empty(shape, ctx, dtype);
        // assert!(e.is_ok());
    }
}
