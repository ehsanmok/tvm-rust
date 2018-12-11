use std::convert::TryFrom;
use std::mem;
use std::os::raw::c_int;
use std::ptr;
use std::slice;

use num_traits::Num;
use rust_ndarray::{Array, ArrayD};

use ts;

use Error;
use Result;
use TVMContext;
use TVMType;

#[derive(Debug)]
pub struct NDArray {
    pub(crate) handle: ts::TVMArrayHandle,
    is_view: bool,
}

impl NDArray {
    pub(crate) fn new(handle: ts::TVMArrayHandle, is_view: bool) -> Self {
        NDArray {
            handle: handle,
            is_view: is_view,
        }
    }

    pub fn as_handle(&self) -> ts::TVMArrayHandle {
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

    // TODO: inline
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
        unsafe {
            (*self.handle).strides.as_mut().map(|pv| {
                let sz = self.ndim();
                let mut v: Vec<usize> = Vec::with_capacity(sz * mem::size_of::<usize>());
                v.as_mut_ptr()
                    .copy_from_nonoverlapping(pv as *mut _ as *const _, sz);
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

    pub fn to_vec<T>(&self) -> Result<Vec<T>> {
        if self.shape().is_none() {
            return Err(Error::EmptyArray);
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

    pub fn to_bytearray(&self) -> Result<Box<[u8]>> {
        self.to_vec::<u8>().map(|v| v.into_boxed_slice())
    }

    // TODO: restrict to i32, u32, f32 as TVMType repr
    pub fn copy_from_buffer<T>(&mut self, data: &mut [T]) {
        check_call!(ts::TVMArrayCopyFromBytes(
            self.handle,
            data.as_ptr() as *mut _,
            data.len() * mem::size_of::<T>()
        ));
    }

    pub fn copy_to_ndarray(&self, target: NDArray) -> Result<NDArray> {
        if self.dtype() != target.dtype() {
            return Err(Error::TypeMismatch {
                expected: self.dtype().to_string(),
                found: target.dtype().to_string(),
            });
        }
        check_call!(ts::TVMArrayCopyFromTo(
            self.handle,
            target.handle,
            ptr::null_mut() as ts::TVMStreamHandle
        ));
        Ok(target)
    }

    pub fn copy_to_ctx(&self, target: &TVMContext) -> Result<NDArray> {
        let tmp = empty(&mut self.shape().unwrap(), target.clone(), self.dtype());
        let copy = self.copy_to_ndarray(tmp)?;
        Ok(copy)
    }

    pub fn from_rust_ndarray<T: Num + Copy>(
        rnd: &ArrayD<T>,
        ctx: TVMContext,
        dtype: TVMType,
    ) -> Result<Self> {
        let mut shape = rnd.shape().to_vec();
        let mut nd = empty(&mut shape, ctx, dtype);
        let mut vec = Array::from_iter(rnd.iter())
            .iter()
            .map(|e| **e)
            .collect::<Vec<T>>();
        nd.copy_from_buffer(vec.as_mut_slice());
        Ok(nd)
    }
}

pub fn empty(shape: &mut Vec<usize>, ctx: TVMContext, dtype: TVMType) -> NDArray {
    let mut handle = ptr::null_mut() as ts::TVMArrayHandle;
    check_call!(ts::TVMArrayAlloc(
        shape.as_ptr() as *const i64,
        shape.len() as c_int,
        dtype.inner.code as c_int,
        dtype.inner.bits as c_int,
        dtype.inner.lanes as c_int,
        ctx.device_type.0 as c_int,
        ctx.device_id as c_int,
        &mut handle as *mut _,
    ));
    NDArray::new(handle, false)
}

macro_rules! impl_from_ndarray_rustndarray {
    ($type:ty, $type_name:tt) => {
        impl<'a> TryFrom<&'a NDArray> for ArrayD<$type> {
            type Error = Error;
            fn try_from(nd: &NDArray) -> Result<ArrayD<$type>> {
                if nd.shape().is_none() {
                    return Err(Error::EmptyArray);
                }
                assert_eq!(nd.dtype(), TVMType::from($type_name), "Type mismatch");
                Ok(
                    Array::from_shape_vec(nd.shape().unwrap().clone(), nd.to_vec::<$type>()?)
                        .unwrap(),
                )
            }
        }

        impl<'a> TryFrom<&'a mut NDArray> for ArrayD<$type> {
            type Error = Error;
            fn try_from(nd: &mut NDArray) -> Result<ArrayD<$type>> {
                if nd.shape().is_none() {
                    return Err(Error::EmptyArray);
                }
                assert_eq!(nd.dtype(), TVMType::from($type_name), "Type mismatch");
                Ok(
                    Array::from_shape_vec(nd.shape().unwrap().clone(), nd.to_vec::<$type>()?)
                        .unwrap(),
                )
            }
        }
    };
}

impl_from_ndarray_rustndarray!(i32, "int");
impl_from_ndarray_rustndarray!(u32, "uint");
impl_from_ndarray_rustndarray!(f32, "float");

impl Drop for NDArray {
    fn drop(&mut self) {
        if !self.is_view {
            check_call!(ts::TVMArrayFree(self.handle));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let mut shape = vec![1, 2, 3];
        let ctx = TVMContext::cpu(0);
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
        let mut data = vec![1i32, 2, 3, 4];
        let ctx = TVMContext::cpu(0);
        let mut ndarray = empty(&mut shape, ctx, TVMType::from("int"));
        assert!(ndarray.to_vec::<i32>().is_ok());
        ndarray.copy_from_buffer(&mut data);
        assert_eq!(ndarray.shape(), Some(shape));
        assert_eq!(ndarray.to_vec::<i32>().unwrap(), data);
        assert_eq!(ndarray.ndim(), 1);
        assert!(ndarray.is_contiguous());
        assert_eq!(ndarray.byte_offset(), 0);
        let mut shape = vec![4];
        let e = empty(&mut shape, TVMContext::cpu(0), TVMType::from("int"));
        let nd = ndarray.copy_to_ndarray(e);
        assert!(nd.is_ok());
        assert_eq!(nd.unwrap().to_vec::<i32>().unwrap(), data);
    }

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err`")]
    fn copy_wrong_dtype() {
        let mut shape = vec![4];
        let mut data = vec![1f32, 2., 3., 4.];
        let ctx = TVMContext::cpu(0); // TVMContext::gpu(0);
        let mut nd_float = empty(&mut shape, ctx.clone(), TVMType::from("float"));
        nd_float.copy_from_buffer(&mut data);
        let empty_int = empty(&mut shape, ctx, TVMType::from("int"));
        nd_float.copy_to_ndarray(empty_int).unwrap();
    }

    #[test]
    fn rust_ndarray() {
        let a = Array::from_shape_vec((2, 2), vec![1f32, 2., 3., 4.])
            .unwrap()
            .into_dyn();
        let nd =
            NDArray::from_rust_ndarray(&a, TVMContext::cpu(0), TVMType::from("float")).unwrap();
        assert_eq!(nd.shape(), Some(vec![2, 2]));
        let rnd: ArrayD<f32> = ArrayD::try_from(&nd).unwrap();
        assert!(rnd.all_close(&a, 1e-8f32));
    }
}
