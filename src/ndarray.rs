extern crate tvm_sys as tvm;

use std::marker::PhantomData;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

pub use TVMContext;
pub use TVMError;
pub use TVMResult;
pub use TVMType;

// look at TVMArrayAlloc args
pub struct NDArray {
    handle: tvm::TVMArrayHandle,
    pub is_view: bool,
    pub dtype: TVMType,
}

impl NDArray {
    pub fn empty(shape: Vec<u32>, dtype: TVMType, ctx: TVMContext) -> TVMResult<Self> {
        let handle = ptr::null_mut();
        check_call!(tvm::TVMArrayAlloc(
            shape.as_ptr() as *const i64,
            shape.len() as c_int,
            dtype.inner.code as c_int,
            dtype.inner.bits as c_int,
            dtype.inner.lanes as c_int,
            ctx.device_type.0 as c_int,
            ctx.device_id,
            handle,
        ));
        Ok(Self::new(handle, false, dtype))
    }

    fn new(handle: *mut tvm::TVMArrayHandle, is_view: bool, dtype: TVMType) -> Self {
        unsafe {
            NDArray {
                handle: *handle,
                is_view: is_view,
                dtype: dtype,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let e = NDArray::empty(vec![1], TVMType::from("float"), TVMContext::cpu(0));
    }
}
