extern crate tvm_sys as tvm;

use std::marker::PhantomData;
use std::mem;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;
use std::slice;
use std::convert::TryFrom;

use rndarray;

use TVMArray;
use TVMContext;
use TVMError;
use TVMResult;
use TVMType;
use TVMTypeCode;
use TypeCode;

pub struct NDArray {
    handle: tvm::TVMArrayHandle,
    pub is_view: bool,
}

impl TVMTypeCode for NDArray {
    fn type_code() -> TypeCode {
        TypeCode {
            sys: tvm::TVMTypeCode::kArrayHandle,
        }
    }
}

pub fn empty(shape: &mut Vec<i64>, ctx: TVMContext, dtype: TVMType) -> NDArray {
    let mut handle = ptr::null_mut() as tvm::TVMArrayHandle;
    check_call!(tvm::TVMArrayAlloc(
        shape.as_ptr() as *const i64,
        shape.len() as c_int,
        dtype.inner.code as c_int,
        dtype.inner.bits as c_int,
        dtype.inner.lanes as c_int,
        ctx.device_type.inner as c_int,
        ctx.device_id as c_int,
        &mut handle as *mut _,
    ));
    NDArray::new(handle, false)
}

impl NDArray {
    fn new(handle: tvm::TVMArrayHandle, is_view: bool) -> Self {
        NDArray {
            handle: handle,
            is_view: is_view,
        }
    }

    pub fn shape(&self) -> Option<Vec<i64>> {
        let arr = unsafe { *(self.handle) };
        if arr.shape.is_null() || arr.data.is_null() {
            return None;
        };
        let slc = unsafe { slice::from_raw_parts_mut(arr.shape, arr.ndim as usize) };
        Some(slc.to_vec())
    }

    pub fn size(&self) -> Option<i64> {
        self.shape().map(|v| v.into_iter().product())
    }

    pub fn ctx(&self) -> TVMContext {
        let dlctx = unsafe { (*self.handle).ctx };
        dlctx.into()
    }

    pub fn context(&self) -> TVMContext {
        self.ctx()
    }

    pub fn dtype(&self) -> TVMType {
        let dldtype = unsafe { (*self.handle).dtype };
        dldtype.into()
    }

    // TODO: add to bluss's ndarray
    pub fn to_vec<T>(&self) -> TVMResult<Vec<T>> {
        if self.shape().is_none() {
            panic!("Cannot copy empty array to Vec");
        }
        let earr = empty(&mut self.shape().unwrap(), TVMContext::cpu(0), self.dtype());
        let target = self.copyto(earr).unwrap();
        let arr = unsafe { *(target.handle) };
        let sz = self.size().unwrap() as usize;
        let mut v: Vec<T> = Vec::with_capacity(sz * mem::size_of::<T>());
        unsafe {
            v.as_mut_ptr().copy_from_nonoverlapping(arr.data as *const T, sz);
            v.set_len(sz);
        }
        Ok(v)
    }

    pub fn copyfrom<T>(&mut self, data: &mut Vec<T>) {
        // lane = 1
        check_call!(tvm::TVMArrayCopyFromBytes(self.handle, data.as_mut_ptr() as *mut _, data.len() * mem::size_of::<T>()));
    }

    pub fn copyto(&self, target: NDArray) -> TVMResult<NDArray> {
        if self.shape().is_none() {
            panic!("Cannot copy empty array to {}", target.context());
        }
        check_call!(tvm::TVMArrayCopyFromTo(self.handle, target.handle, ptr::null_mut() as *mut _));
        Ok(target)
    }
}

// TODO: bluss's ndarray support
//impl<'a> TryFrom<&'a NDArray> for rndarray::ArrayD<T> {
//    type Error = TVMError;
//    fn try_from(array: &NDArray) -> TVMResult<rndarray::ArrayD<T>> {
//        unimplemented!()
//    }
//}


impl Drop for NDArray {
    fn drop(&mut self) {
        check_call!(tvm::TVMArrayFree(self.handle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut shape = vec![1, 2, 3];
        let ctx = TVMContext::cpu(0); // TVMContext::opencl(0);
        let ndarray = empty(&mut shape, ctx, TVMType::from("int"));
        assert_eq!(ndarray.shape().unwrap(), shape);
        assert_eq!(ndarray.size().unwrap(), shape.into_iter().product());
    }

    #[test]
    fn copy() {
        let mut shape = vec![1, 4];
        let mut data = vec![1f32, 2., 3., 4.];
        // let mut data = vec![1i32, 2, 3, 4];
        let ctx = TVMContext::cpu(0); // TVMContext::gpu(0);
        let mut ndarray = empty(&mut shape, ctx, TVMType::from("float"));
        ndarray.copyfrom(&mut data);
        assert_eq!(ndarray.to_vec::<f32>().unwrap(), data);
    }
}
