use std::convert::TryFrom;
use std::ffi::CStr;
use std::marker::Send;
use std::mem;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::slice;
use std::str;

use tvm;

use TVMError;
use TVMResult;
use TVMTypeCode;
use TVMValue;
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

    pub fn get_function(name: &'static str, allow_missing: bool) -> Option<Function> {
        let name = name.to_owned();
        list_global_func_names()
            .into_iter()
            .find(move |s| *s == name)
            .map(|nm| get_global_func(&nm, allow_missing).unwrap())
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
        TypeCode::kFuncHandle
    }
}

impl FnOnce<(Vec<TVMArgValue>,)> for Function {
    type Output = TVMRetValue;
    extern "rust-call" fn call_once(self, mut args: (Vec<TVMArgValue>,)) -> Self::Output {
        let mut ret_val: tvm::TVMValue = tvm::TVMValue { v_int64: 0 };
        let mut ret_type_code = 0 as c_int;
        check_call!(tvm::TVMFuncCall(
            self.handle,
            args.0.as_mut_ptr() as *mut _,
            args.0.as_mut_ptr() as *mut _,
            args.0.len() as c_int,
            &mut ret_val as *mut tvm::TVMValue,
            &mut ret_type_code as *mut _
        ));
        TVMRetValue::new(TVMValue { inner: ret_val }, TypeCode::from(&ret_type_code))
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        check_call!(tvm::TVMFuncFree(self.handle));
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
        mem::forget(name); // fixes --test-threads=1 !?!
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

// TODO: make a Rust fn into callback
#[derive(Debug, Clone, Hash)]
pub struct TVMArgValue {
    value: TVMValue,
    type_code: TypeCode,
}

pub type TVMRetValue = TVMArgValue;

impl TVMArgValue {
    pub fn new(value: TVMValue, type_code: TypeCode) -> Self {
        TVMArgValue { value, type_code }
    }
}

#[derive(Debug, Clone)]
pub struct PackedFunc {
    inner: tvm::TVMPackedCFunc,
}

impl PackedFunc {
    pub fn register(&self, name: &'static str, override_: bool) -> TVMResult<()> {
        let name = name.to_owned();
        let func = Function::from(self);
        let override_ = if override_ { 1 } else { 0 };
        check_call!(tvm::TVMFuncRegisterGlobal(
            name.as_ptr() as *const c_char,
            func.handle,
            override_ as c_int,
        ));
        Ok(())
    }
}

impl<'a> From<&'a PackedFunc> for Function {
    fn from(packed_func: &PackedFunc) -> Self {
        let mut fhandle = ptr::null_mut() as tvm::TVMFunctionHandle;
        let resource_handle = ptr::null_mut() as *mut c_void;
        check_call!(tvm::TVMFuncCreateFromCFunc(
            packed_func.inner,
            resource_handle,
            None,
            &mut fhandle as *mut _
        ));
        Self::new(fhandle, false)
    }
}
// TODO: impl Fn, FnMut
impl FnOnce<(Vec<TVMArgValue>,)> for PackedFunc {
    type Output = TVMRetValue;
    extern "rust-call" fn call_once(self, args: (Vec<TVMArgValue>,)) -> Self::Output {
        let func = Function::from(&self);
        func(args.0)
    }
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
        assert!(Function::get_function("tvm.graph_runtime.create", false).is_some());
        assert!(Function::get_function("does not exists!", false).is_none());
    }

    #[test]
    fn register_fn() {
        unsafe extern "C" fn zero_fn(
            args: *mut tvm::TVMValue,
            type_codes: *mut ::std::os::raw::c_int,
            num_args: ::std::os::raw::c_int,
            ret: tvm::TVMRetValueHandle,
            resource_handle: *mut ::std::os::raw::c_void,
        ) -> ::std::os::raw::c_int {
            0
        }
        let zero_packed: PackedFunc = PackedFunc {
            inner: Some(zero_fn),
        };
        let reg = zero_packed.register("zero_fn", false);
        assert!(reg.is_ok());
        //let arg = TVMArgValue::new(TVMValue::from(&0i64), TypeCode::from(&0i64));
        //println!("{:?}", arg);
        // println!("{:?}", zero_packed(vec!(arg)));
        // let ret = zero_packed(vec![arg]);
        // println!("{:?}", ret.type_code);
    }
}
