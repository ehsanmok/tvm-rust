use std::convert::TryFrom;
use std::ffi::CStr;
use std::marker::Send;
use std::mem;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::slice;
use std::str;

use tvm;

use super::*;

#[derive(Debug, Clone, Hash)]
pub struct Function<'a> {
    handle: Box<tvm::TVMFunctionHandle>,
    is_global: bool,
    is_released: bool,
    pub(crate) arg_buf: Option<Box<[TVMArgValue<'a>]>>,
}

impl<'a> Function<'a> {
    pub fn new(
        handle: Box<tvm::TVMFunctionHandle>,
        is_global: bool,
        is_released: bool,
        arg_buf: Option<Box<[TVMArgValue<'a>]>>,
    ) -> Self {
        Function {
            handle: handle,
            is_global: is_global,
            is_released: is_released,
            arg_buf: arg_buf,
        }
    }

    pub fn get_function(
        name: String,
        is_global: bool,
        allow_missing: bool,
    ) -> Option<Function<'a>> {
        // let name = name.to_owned();
        list_global_func_names()
            .into_iter()
            .find(move |s| *s == &name)
            .map(|nm| get_global_func(&nm, is_global, allow_missing).unwrap())
    }

    pub fn as_handle(&self) -> Box<tvm::TVMFunctionHandle> {
        self.handle.clone()
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }

    pub fn push_arg<'b: 'a, T: 'b + ?Sized>(mut self, arg: &'b T) -> Self
    where
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        let tvm_arg = TVMArgValue::new(TVMValue::from(arg), TypeCode::from(arg));
        //println!("{:?}", tvm_arg);
        if self.arg_buf.is_none() {
            self.arg_buf = Some(Box::new([tvm_arg]));
            return self;
        } else {
            let new_arg_buf = self.arg_buf.take().map(|bbuf| {
                let mut new_buf = Vec::with_capacity(bbuf.len() + 1);
                let tmp = Vec::from(bbuf);
                for elt in tmp {
                    new_buf.push(elt);
                }
                new_buf.push(tvm_arg);
                //println!("{:?}", new_buf);
                new_buf.into_boxed_slice()
            });
            Self::new(
                self.handle.clone(),
                self.is_global,
                self.is_released,
                new_arg_buf,
            )
        }
    }

    pub fn invoke(self) -> TVMRetValue<'a> {
        self(())
    }
}

impl<'a> FnOnce<((),)> for Function<'a> {
    type Output = TVMRetValue<'a>;
    extern "rust-call" fn call_once(self, _: ((),)) -> Self::Output {
        let mut ret_val = tvm::TVMValue { v_int64: 0 };
        let mut ret_type_code = 0;
        let arg_buf = self.arg_buf.clone().unwrap();
        let num_args = arg_buf.len();
        let mut values = arg_buf
            .iter()
            .map(|tav| tav.clone().value.inner)
            .collect::<Vec<tvm::TVMValue>>();
        values.truncate(num_args);

        let mut tcodes = arg_buf
            .iter()
            .map(|tav| tav.clone().type_code as c_int)
            .collect::<Vec<_>>();
        tcodes.truncate(num_args);

        check_call!(tvm::TVMFuncCall(
            *self.handle,
            values.as_mut_ptr(),
            tcodes.as_mut_ptr(),
            num_args as c_int,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _
        ));
        TVMRetValue::new(
            TVMValue::new(ValueKind::Return, ret_val),
            TypeCode::from(&ret_type_code),
        )
    }
}

impl<'a> Drop for Function<'a> {
    fn drop(&mut self) {
        if !self.is_released {
            if !self.is_global {
                check_call!(tvm::TVMFuncFree(*self.handle));
                self.arg_buf.take();
                //unsafe { ptr::drop_in_place(*self.handle) };
                self.is_released = true;
            }
        }
    }
}

fn get_global_func(name: &str, is_global: bool, allow_missing: bool) -> Option<Function> {
    let name = name.to_owned();
    let mut handle = ptr::null_mut() as tvm::TVMFunctionHandle;
    check_call!(tvm::TVMFuncGetGlobal(
        name.as_ptr() as *const c_char,
        &mut handle as *mut _
    ));
    if !(handle.is_null()) {
        mem::forget(name);
        return Some(Function::new(Box::new(handle), is_global, false, None));
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
    // TODO: posssible thread sanitization!
    #[test]
    fn list_global_func() {
        let list = list_global_func_names();
        //println!("{:?}", list);
        assert!(
            list.iter()
                .find(|ref s| ***s == "tvm.graph_runtime.create")
                .is_some()
        );
    }

    #[test]
    fn global_func() {
        assert!(get_global_func("tvm.graph_runtime.create", true, false).is_some());
    }

    #[test]
    fn get_fn() {
        assert!(
            Function::get_function("tvm.graph_runtime.remote_create".to_owned(), true, false)
                .is_some()
        );
        assert!(Function::get_function("does not exists!".to_owned(), false, false).is_none());
    }

    #[test]
    fn arg() {
        let mut func =
            Function::get_function("tvm.graph_runtime.remote_create".to_owned(), true, false)
                .unwrap();
        func = func.push_arg(&10).push_arg("test");
        //println!("{:?}", func.arg_buf);
        assert!(func.arg_buf.is_some());
        assert_eq!(func.arg_buf.take().map(|bv| Vec::from(bv).len()), Some(2));
    }
}
