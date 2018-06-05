extern crate tvm_sys as tvm;

use std::marker::PhantomData;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;

// pub use DLTensor as TVMArray;
pub use TVMContext;
pub use TVMError;
pub use TVMResult;
pub use TVMType;

// look at TVMArrayAlloc args
pub struct NDArray {
    handle: *mut tvm::TVMArrayHandle,
    pub is_view: bool,
    pub dtype: TVMType,
}

impl NDArray {
    pub fn empty(shape: &[i32], dtype: TVMType, ctx: TVMContext) -> TVMResult<Self> {
        // let empty = tvm::DLTensor {
        //     data: ptr::null_mut() as *mut c_void,
        //     ctx: tvm::DLContext::from(ctx),
        //     ndim: shape.len() as c_int,
        //     dtype: tvm::DLDataType::from(dtype),
        //     shape: shape.as_ptr() as *mut i64,
        //     strides: 1,
        //     byte_offset: 0u64,
        // };
        //let handle = &empty as *mut tvm::TVMArrayHandle;
        let handle = ptr::null_mut() as *mut tvm::TVMArrayHandle;
        println!("{:?}", handle);
        let res = unsafe {
            tvm::TVMArrayAlloc(
                shape.as_ptr() as *const i64,
                shape.len() as c_int,
                dtype.inner.code as c_int,
                dtype.inner.bits as c_int,
                dtype.inner.lanes as c_int,
                ctx.device_type.inner as c_int,
                ctx.device_id,
                handle,
            )
        };
        println!("{:?}", handle);
        // check_call!(tvm::TVMArrayAlloc(
        //     shape.as_ptr() as *const i64,
        //     shape.len() as c_int,
        //     dtype.inner.code as c_int,
        //     dtype.inner.bits as c_int,
        //     dtype.inner.lanes as c_int,
        //     ctx.device_type.0 as c_int,
        //     ctx.device_id,
        //     handle,
        // ));
        Ok(Self::new(handle, false, dtype))
    }

    fn new(handle: *mut tvm::TVMArrayHandle, is_view: bool, dtype: TVMType) -> Self {
        NDArray {
            handle: handle,
            is_view: is_view,
            dtype: dtype,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        let shape = &[1, 2];
        let dtype = TVMType::from("float");
        let ctx = TVMContext::cpu(0);
        let e = NDArray::empty(shape, dtype, ctx);
        assert!(e.is_ok());
    }
}
