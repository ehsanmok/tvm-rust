use std::ffi::CStr;
use std::mem;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::slice;
use std::str;

use tvm;

use TVMTypeCode;
use TypeCode;

#[derive(Debug, Clone, Hash)]
pub struct Function {
    handle: tvm::TVMFunctionHandle, // *mut c_void
    is_global: bool,
}

impl Function {
    pub fn new(handle: tvm::TVMFunctionHandle, is_global: bool) -> Self {
        Function {
            handle: handle,
            is_global: is_global,
        }
    }

    pub fn get_function(name: &str) -> Option<Function> {
        let name = name.to_owned();
        list_global_func_names()
            .into_iter()
            .find(move |s| *s == name)
            .map(|nm| get_global_func(&nm, false).unwrap())
    }

    pub fn as_handle(&self) -> tvm::TVMFunctionHandle {
        self.handle
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }
}

impl TVMTypeCode for Function {
    fn type_code() -> TypeCode {
        TypeCode {
            sys: tvm::TVMTypeCode::kFuncHandle,
        }
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        check_call!(tvm::TVMFuncFree(self.handle));
        mem::drop(self);
    }
}

fn get_global_func(name: &str, allow_missing: bool) -> Option<Function> {
    let name = name.to_owned();
    let mut handle = ptr::null_mut() as tvm::TVMFunctionHandle;
    check_call!(tvm::TVMFuncGetGlobal(
        name.as_ptr() as *const c_char,
        &mut handle as *mut _
    ));
    if !(handle.is_null()) {
        return Some(Function::new(handle, false));
    } else {
        if allow_missing {
            return None;
        } else {
            panic!("Cannot find global function {}", name);
        }
    }
}

fn list_global_func_names() -> Vec<&'static str> {
    let mut out_size = 0 as c_int;
    let mut name = ptr::null() as *const c_char;
    let mut out_array = &mut name as *mut _;
    check_call!(tvm::TVMFuncListGlobalNames(
        &mut out_size as *mut _,
        &mut out_array as *mut _
    ));
    let list = unsafe { slice::from_raw_parts(out_array, out_size as usize) };
    list.iter()
        .map(|&p| unsafe { CStr::from_ptr(p) })
        .map(|cs| cs.to_bytes())
        .map(|bs| str::from_utf8(bs).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_func() {
        assert!(get_global_func("tvm.graph_runtime.create", false).is_some());
    }

    #[test]
    fn list_global_func() {
        let list = list_global_func_names();
        // println!("{:?}", list);
        assert!(
            list.iter()
                .find(|ref s| ***s == "tvm.graph_runtime.create")
                .is_some()
        );
    }

    #[test]
    fn get_fn() {
        assert!(Function::get_function("tvm.graph_runtime.create").is_some());
        assert!(Function::get_function("does not exists!").is_none());
    }
}
