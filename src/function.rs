extern crate tvm_sys as tvm;

use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use TVMTypeCode;
pub use TypeCode;

#[derive(Debug, Clone)]
pub struct Function {
    handle: tvm::TVMFunctionHandle, // *mut c_void
    is_resident: bool,
}

impl Function {
    pub fn new(handle: tvm::TVMFunctionHandle, is_resident: bool) -> Self {
        Function {
            handle: handle,
            is_resident: is_resident,
        }
    }
}

impl Default for Function {
    fn default() -> Self {
        Function {
            handle: ptr::null_mut() as tvm::TVMFunctionHandle,
            is_resident: false,
        }
    }
}

impl TVMTypeCode for Function {
    fn type_code() -> TypeCode {
        TypeCode { sys: tvm::TVMTypeCode::kFuncHandle }
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        check_call!(tvm::TVMFuncFree(self.handle));
    }
}

pub fn get_global_func(
    name: &str,
    is_resident: bool,
    allow_missing: bool,
) -> Option<Function> {
    let mut handle = ptr::null_mut() as tvm::TVMFunctionHandle;
    let out: *mut tvm::TVMFunctionHandle = &mut handle as *mut tvm::TVMFunctionHandle;
    check_call!(tvm::TVMFuncGetGlobal(name.as_ptr() as *const c_char, out));
    if !out.is_null() {
        Some(Function::new(handle, is_resident))
    } else {
        if allow_missing {
            return None;
        } else {
            panic!("Cannot find global function {}", name);
        }
    }
}

pub fn list_global_func_names() -> Vec<&'static str> {
    // let mut list: Vec<&'static str> = Vec::new();
    let out_size = ptr::null_mut() as *mut c_int;
    let out_array = ptr::null_mut() as *mut *mut *const c_char;
    check_call!(tvm::TVMFuncListGlobalNames(out_size, out_array));
    let size = unsafe { *out_size };
    let mut list: Vec<&'static str> = Vec::with_capacity(size as usize);
    // TODO: fill out the list
    unimplemented!()
}
