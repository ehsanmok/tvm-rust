//! Provides [`TVMByteArray`] which is used for passing the model parameters
//! (stored as byte-array) to a runtime module.
//!
//! This function can be obtained from a graph runtime module loading the model.
//! For more detail, please see the example `resnet` in `examples` repository.

use std::ffi::CString;
use std::mem;

use ts;

/// A struct holding the TVM byte-array.
///
/// ## Example
///
/// ```
/// let v = b"hello".to_vec();
/// let barr = TVMByteArray::from(&v);
/// assert_eq!(barr.len(), v.len());
/// assert_eq!(barr.data(), vec![104i8, 101, 108, 108, 111]);
/// ```
#[derive(Debug, Clone)]
pub struct TVMByteArray {
    pub(crate) inner: ts::TVMByteArray,
}

impl TVMByteArray {
    pub(crate) fn new(barr: ts::TVMByteArray) -> TVMByteArray {
        TVMByteArray { inner: barr }
    }

    /// Gets the length of the underlying byte-array
    pub fn len(&self) -> usize {
        self.inner.size
    }

    /// Gets the underlying byte-array as `Vec<i8>`
    pub fn data(&self) -> Vec<i8> {
        unsafe {
            let sz = self.len();
            let mut ret_buf = Vec::with_capacity(sz);
            ret_buf.set_len(sz);
            self.inner.data.copy_to(ret_buf.as_mut_ptr(), sz);
            ret_buf
        }
    }
}

impl<'a> From<&'a Vec<u8>> for TVMByteArray {
    fn from(arg: &Vec<u8>) -> Self {
        unsafe {
            let data = CString::from_vec_unchecked(arg.to_vec());
            let barr = ts::TVMByteArray {
                data: data.as_ptr(),
                size: arg.len(),
            };
            mem::forget(data);
            TVMByteArray::new(barr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        let v = vec![1u8, 2, 3];
        let barr = TVMByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), vec![1i8, 2, 3]);
        let v = b"hello".to_vec();
        let barr = TVMByteArray::from(&v);
        assert_eq!(barr.len(), v.len());
        assert_eq!(barr.data(), vec![104i8, 101, 108, 108, 111]);
    }
}
