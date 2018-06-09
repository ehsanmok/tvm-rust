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
    pub dtype: TVMType, // TODO: remove
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

    pub fn empty(shape: &mut Vec<i64>, ctx: TVMContext, dtype: TVMType) -> Self {
        let mut handle = ptr::null_mut() as tvm::TVMArrayHandle;
        let out = &mut handle as *mut tvm::TVMArrayHandle;
        check_call!(tvm::TVMArrayAlloc(
            shape.as_ptr() as *const i64,
            shape.len() as c_int,
            dtype.inner.code as c_int,
            dtype.inner.bits as c_int,
            dtype.inner.lanes as c_int,
            ctx.device_type.inner as c_int,
            ctx.device_id as c_int,
            out,
        ));
        Self::new(unsafe { *out }, false, dtype)
    }

    pub fn shape(&self) -> Option<Vec<i64>> {
        let arr = unsafe { *(self.handle) };
        if arr.shape.is_null() { return None };
        let slice = unsafe { ::std::slice::from_raw_parts_mut(arr.shape, arr.ndim as usize) };
        Some(slice.to_vec())
    }

    pub fn size(&self) -> Option<i64> {
        self.shape().map(|v| v.into_iter().product())
    }

    pub fn context(&self) -> TVMContext {
        // TODO: add from for DLContext with ints
        unimplemented!()
    }

    pub fn copyfrom(&mut self, data: &mut Vec<f32>) {
        // lane = 1
        // NDArray, (layer blus's ndarray) and Vec
        let handle = self.handle; // ensure is not null
        let dptr = data.as_mut_ptr() as *mut c_void;
        let nbytes = data.len() * mem::size_of::<f32>();
        check_call!(tvm::TVMArrayCopyFromBytes(handle, dptr, nbytes));
    }

//    pub fn data(&self) -> Option<Vec<f32>> {
//        let arr = unsafe { *(self.handle) };
//        if arr.shape.is_null() { return None };
//        let slice = unsafe { ::std::slice::from_raw_parts_mut(arr.data, arr.ndim as usize) };
//        Some(slice.to_vec())
//    }
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
        let mut shape = vec![1, 2];
        let dtype = TVMType::from("float");
        let ctx = TVMContext::cpu(0);
        let e = NDArray::empty(&mut shape, ctx, dtype);
        // assert!(e.is_ok());
    }

    #[test]
    fn basics() {
        let mut shape = vec![1, 2, 3];
        let ndarray = NDArray::empty(&mut shape, TVMContext::cpu(0), TVMType::from("int"));
        // let ndarray = NDArray::empty(&mut shape, TVMContext::gpu(0), TVMType::from("int"));
        // let ndarray = NDArray::empty(&mut shape, TVMContext::opencl(0), TVMType::from("int"));
        assert_eq!(ndarray.shape().unwrap(), shape);
        assert_eq!(ndarray.size().unwrap(), shape.into_iter().product());
    }

    #[test]
    fn copyfrom() {
        let mut shape = vec![1];
        let mut ndarray = NDArray::empty(&mut shape, TVMContext::cpu(0), TVMType::from("int"));
        ndarray.copyfrom(&mut vec![1.0]);
        // TODO
        // assert_eq!(ndarray.data(), vec![1.0]);
    }
}
