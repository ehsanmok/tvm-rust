use std::convert::TryFrom;
use std::marker::PhantomData;
use std::mem;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::slice;

use rust_ndarray;
use rust_ndarray::ArrayD;
use num_traits::Num;

use tvm;

use super::*;

#[derive(Debug)]
pub struct NDArray {
    pub(crate) handle: tvm::TVMArrayHandle,
    is_view: bool,
}

pub fn empty(shape: &mut Vec<usize>, ctx: TVMContext, dtype: TVMType) -> NDArray {
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

    pub fn as_handle(&self) -> tvm::TVMArrayHandle {
        self.handle
    }

    pub fn is_view(&self) -> bool {
        self.is_view
    }

    pub fn shape(&self) -> Option<Vec<usize>> {
        let arr = unsafe { *(self.handle) };
        if arr.shape.is_null() || arr.data.is_null() {
            return None;
        };
        let slc = unsafe { slice::from_raw_parts_mut(arr.shape as *mut usize, arr.ndim as usize) };
        Some(slc.to_vec())
    }

    pub fn size(&self) -> Option<usize> {
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

    pub fn ndim(&self) -> usize {
        unsafe { (*self.handle).ndim as usize }
    }

    pub fn strides(&self) -> Option<Vec<usize>> {
        unsafe { (*self.handle).strides.as_mut().map(|pv| {
            let sz = self.ndim();
            let mut v: Vec<usize> = Vec::with_capacity(sz * mem::size_of::<usize>());
            v.as_mut_ptr().copy_from_nonoverlapping(pv as *mut _ as *const _, sz);
            v.set_len(sz);
            v
            })
        }
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides().is_none()
    }

    pub fn byte_offset(&self) -> isize {
        unsafe { (*self.handle).byte_offset as isize }
    }

    pub fn to_vec<T>(&self) -> TVMResult<Vec<T>> {
        if self.shape().is_none() {
            panic!("Cannot copy empty array to Vec");
        }
        let earr = empty(&mut self.shape().unwrap(), TVMContext::cpu(0), self.dtype());
        let target = self.copy_to_ndarray(earr).unwrap();
        let arr = unsafe { *(target.handle) };
        let sz = self.size().unwrap() as usize;
        let mut v: Vec<T> = Vec::with_capacity(sz * mem::size_of::<T>());
        unsafe {
            v.as_mut_ptr()
                .copy_from_nonoverlapping(arr.data as *const T, sz);
            v.set_len(sz);
        }
        Ok(v)
    }

    pub fn copy_from_buffer<T>(&mut self, data: &mut [T]) {
        check_call!(tvm::TVMArrayCopyFromBytes(
            self.handle,
            data.as_ptr() as *mut _,
            data.len() * mem::size_of::<T>()
        ));
    }

    pub fn copy_from_ndarray(&mut self, arr: NDArray) {
        self.copy_to_ndarray(arr).unwrap();
    }

    pub fn copy_to_ndarray(&self, target: NDArray) -> TVMResult<NDArray> {
        // TODO: add fmt to TVMType
        assert_eq!(self.dtype(), target.dtype(), "Copy to ndarray expects dtype {:?}, but given {:?}", self.dtype(), target.dtype());
        check_call!(tvm::TVMArrayCopyFromTo(
            self.handle,
            target.handle,
            ptr::null_mut() as *mut _
        ));
        Ok(target)
    }

    pub fn from_rust_ndarray<T: Num + Copy>(
        rnd: &ArrayD<T>,
        ctx: TVMContext,
        dtype: TVMType,
    ) -> TVMResult<Self> {
        let mut shape = rnd.shape().to_vec();
        let mut nd = empty(&mut shape, ctx, dtype);
        let mut vec = rust_ndarray::Array::from_iter(rnd.iter())
            .iter()
            .map(|e| **e)
            .collect::<Vec<T>>();
        nd.copy_from_buffer(vec.as_mut_slice());
        Ok(nd)
    }
}

// TODO: generic?
impl<'a> TryFrom<&'a NDArray> for ArrayD<f32> {
    type Error = TVMError;
    fn try_from(array: &NDArray) -> TVMResult<ArrayD<f32>> {
        if array.shape().is_none() {
            panic!("Cannot convert from empty array");
        }
        assert_eq!(array.dtype(), TVMType::from("float"), "Conversion from Rust ndarray float32 is allowed");
        Ok(rust_ndarray::Array::from_shape_vec(
            array.shape().unwrap().clone(),
            array.to_vec::<f32>().unwrap(),
        ).unwrap())
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
    fn basics() {
        let mut shape = vec![1, 2, 3];
        let ctx = TVMContext::cpu(0); // TVMContext::opencl(0);
        let ndarray = empty(&mut shape, ctx, TVMType::from("int"));
        assert_eq!(ndarray.shape().unwrap(), shape);
        assert_eq!(ndarray.size().unwrap(), shape.into_iter().product());
        assert_eq!(ndarray.ndim(), 3);
        assert!(ndarray.strides().is_none());
        assert_eq!(ndarray.byte_offset(), 0);
    }

    #[test]
    fn copy() {
        let mut shape = vec![4];
        let mut data = vec![1f32, 2., 3., 4.];
        // let mut data = vec![1i32, 2, 3, 4];
        let ctx = TVMContext::cpu(0); // TVMContext::gpu(0);
        let mut ndarray = empty(&mut shape, ctx, TVMType::from("float"));
        ndarray.copy_from_buffer(&mut data);
        assert_eq!(ndarray.shape(), Some(shape));
        assert_eq!(ndarray.to_vec::<f32>().unwrap(), data);
        assert_eq!(ndarray.ndim(), 1);
        assert!(ndarray.is_contiguous());
        assert_eq!(ndarray.byte_offset(), 0);

    }

    #[test]
    #[should_panic]
    fn copy_wrong_dtype() {
        let mut shape = vec![4];
        let mut data = vec![1f32, 2., 3., 4.];
        let ctx = TVMContext::cpu(0); // TVMContext::gpu(0);
        let mut nd_float = empty(&mut shape, ctx.clone(), TVMType::from("float"));
        nd_float.copy_from_buffer(&mut data);
        let mut empty_int = empty(&mut shape, ctx, TVMType::from("int"));
        empty_int.copy_from_ndarray(nd_float);
    }

    #[test]
    fn rust_ndarray() {
        let a = rust_ndarray::Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
            .unwrap()
            .into_dyn();
        let nd =
            NDArray::from_rust_ndarray(&a, TVMContext::cpu(0), TVMType::from("float")).unwrap();
        println!("ndim: {}, strides: {:?}, byte_offset: {}", nd.ndim(), nd.strides(), nd.byte_offset());
        assert_eq!(nd.shape(), Some(vec![2, 2]));
        let rnd = rust_ndarray::ArrayD::try_from(&nd).unwrap();
        assert!(rnd.all_close(&a, 1e-8f32));
    }
}
