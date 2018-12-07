use std::ffi::{CStr, CString};
use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::slice;
use std::str;
use std::sync::Mutex;

use ts;

use Error;
use Module;
use Result;
use TVMArgValue;
use TVMRetValue;
use TVMValue;
use TypeCode;
use ValueKind;

lazy_static! {
    static ref GLOBAL_FUNCTION_NAMES: Mutex<Vec<&'static str>> = list_global_func_names();
}

fn list_global_func_names() -> Mutex<Vec<&'static str>> {
    let mut out_size = 0 as c_int;
    let mut name = ptr::null() as *const c_char;
    let mut out_array = &mut name as *mut _;
    check_call!(ts::TVMFuncListGlobalNames(
        &mut out_size as *mut _,
        &mut out_array as *mut _
    ));
    let list = unsafe { slice::from_raw_parts(out_array, out_size as usize) };
    let list = list
        .iter()
        .map(|&p| unsafe { CStr::from_ptr(p) })
        .map(|cs| cs.to_bytes())
        .map(|bs| str::from_utf8(bs).unwrap())
        .collect();
    Mutex::new(list)
}

pub fn get_global_func(name: &str, is_global: bool, allow_missing: bool) -> Option<Function> {
    let name = CString::new(name).unwrap();
    let mut handle = ptr::null_mut() as ts::TVMFunctionHandle;
    check_call!(ts::TVMFuncGetGlobal(
        name.as_ptr() as *const c_char,
        &mut handle as *mut _
    ));
    if !(handle.is_null()) {
        mem::forget(name);
        return Some(Function::new(handle, is_global, false));
    } else {
        if allow_missing {
            return None;
        } else {
            panic!("Cannot find global function {:?}", name);
        }
    }
}

#[derive(Debug, Clone, Hash)]
pub struct Function {
    handle: ts::TVMFunctionHandle,
    is_global: bool,
    is_released: bool,
}

impl Function {
    pub fn new(handle: ts::TVMFunctionHandle, is_global: bool, is_released: bool) -> Self {
        Function {
            handle: handle,
            is_global: is_global,
            is_released: is_released,
        }
    }

    pub fn get_function(name: String, is_global: bool, allow_missing: bool) -> Option<Function> {
        GLOBAL_FUNCTION_NAMES
            .lock()
            .unwrap()
            .iter()
            .find(|&&s| s == &name)
            .map(|nm| get_global_func(nm, is_global, allow_missing).unwrap())
    }

    pub fn as_handle(&self) -> ts::TVMFunctionHandle {
        self.handle
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }

    pub fn is_released(&self) -> bool {
        self.is_released
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        if !self.is_released {
            if !self.is_global {
                check_call!(ts::TVMFuncFree(self.handle));
                self.is_released = true;
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Builder<'a> {
    pub func: Option<Function>,
    pub arg_buf: Option<Box<[TVMArgValue<'a>]>>,
    pub ret_buf: Option<Box<[TVMRetValue<'a>]>>,
}

impl<'a> Builder<'a> {
    pub fn new(
        func: Option<Function>,
        arg_buf: Option<Box<[TVMArgValue<'a>]>>,
        ret_buf: Option<Box<[TVMRetValue<'a>]>>,
    ) -> Self {
        Self {
            func,
            arg_buf,
            ret_buf,
        }
    }

    pub fn get_function(
        &mut self,
        name: String,
        is_global: bool,
        allow_missing: bool,
    ) -> &mut Self {
        self.func = Function::get_function(name, is_global, allow_missing);
        self
    }

    pub fn arg<'b, T: ?Sized>(&mut self, arg: &'b T) -> &mut Self
    where
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        let tvm_arg = TVMArgValue::from(arg);
        if self.arg_buf.is_none() {
            self.arg_buf = Some(Box::new([tvm_arg]));
        } else {
            let new_arg_buf = self.arg_buf.take().map(|bbuf| {
                let mut new_arg_buf = Vec::from(bbuf);
                new_arg_buf.push(tvm_arg);
                let new_len = new_arg_buf.len();
                new_arg_buf.truncate(new_len);
                new_arg_buf.into_boxed_slice()
            });
            self.arg_buf = new_arg_buf;
        }
        self
    }

    pub fn args<'b, T: 'b + ?Sized, I>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = &'b T>,
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        for arg in args {
            self.arg(&arg);
        }
        self
    }

    pub fn accept_ret<'b, T: 'b + ?Sized>(&mut self, arg: &'b mut T) -> &mut Self
    where
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        let tvm_ret = TVMRetValue::new(TVMValue::from(arg), TypeCode::from(arg));
        if self.ret_buf.is_none() {
            self.ret_buf = Some(Box::new([tvm_ret]));
        } else {
            let new_ret_buf = self.ret_buf.take().map(|bbuf| {
                let mut new_buf = Vec::with_capacity(1);
                new_buf.push(tvm_ret);
                new_buf.into_boxed_slice()
            });
            self.arg_buf = new_ret_buf;
        }
        self
    }

    pub fn invoke(&mut self) -> Result<TVMRetValue<'a>> {
        self.clone()(())
    }
}

impl<'a> FnOnce<((),)> for Builder<'a> {
    type Output = Result<TVMRetValue<'a>>;
    extern "rust-call" fn call_once(self, _: ((),)) -> Self::Output {
        if self.func.is_none() {
            return Err(Error::NoFunction);
        }
        let mut ret_val = unsafe { mem::uninitialized::<ts::TVMValue>() };
        let mut ret_type_code = 0 as c_int;
        let arg_buf = self.arg_buf.clone().unwrap();
        let mut num_args = arg_buf.len();
        let mut values = arg_buf
            .iter()
            .map(|tav| tav.clone().value.inner)
            .collect::<Vec<ts::TVMValue>>();
        let mut tcodes = arg_buf
            .iter()
            .map(|tav| tav.clone().type_code as c_int)
            .collect::<Vec<_>>();

        if self.ret_buf.is_some() {
            num_args = num_args + 1;
            ret_val = *self.ret_buf.clone().unwrap()[0].value;
            ret_type_code = self.ret_buf.clone().unwrap()[0].type_code as c_int;
            values.append(&mut vec![ret_val]);
            tcodes.append(&mut vec![ret_type_code]);
        }
        values.truncate(num_args);
        tcodes.truncate(num_args);
        check_call!(ts::TVMFuncCall(
            self.func.unwrap().handle,
            values.as_mut_ptr(),
            tcodes.as_mut_ptr(),
            num_args as c_int,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _
        ));
        let ret = TVMRetValue::new(
            TVMValue::new(ValueKind::Return, ret_val),
            ret_type_code.into(),
        );
        Ok(ret)
    }
}

impl<'a> From<Function> for Builder<'a> {
    fn from(func: Function) -> Self {
        Builder::new(Some(func), None, None)
    }
}

impl<'a: 'b, 'b> From<&'b mut Module> for Builder<'a> {
    fn from(module: &mut Module) -> Self {
        Builder::new(module.entry.take(), None, None)
    }
}

unsafe extern "C" fn tvm_callback(
    args: *mut ts::TVMValue,
    type_codes: *mut c_int,
    num_args: c_int,
    ret: ts::TVMRetValueHandle,
    fhandle: *mut c_void,
) -> c_int {
    let len = num_args as usize;
    let args_list = unsafe { slice::from_raw_parts_mut(args, len).to_vec() };
    let type_codes_list = unsafe { slice::from_raw_parts_mut(type_codes, len).to_vec() };
    let mut local_args: Vec<TVMArgValue> = Vec::new();
    let mut value = unsafe { mem::uninitialized::<ts::TVMValue>() };
    let mut tcode = unsafe { mem::uninitialized::<c_int>() };
    let rust_fn = unsafe {
        mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue<'static>>>(fhandle)
    };
    for i in 0..len {
        value = args_list[i];
        tcode = type_codes_list[i];
        if tcode == TypeCode::kNodeHandle as c_int
            || tcode == TypeCode::kFuncHandle as c_int
            || tcode == TypeCode::kModuleHandle as c_int
        {
            check_call!(ts::TVMCbArgToReturn(&mut value as *mut _, tcode));
        }
        local_args.push(TVMArgValue::new(
            TVMValue::new(ValueKind::Handle, value),
            tcode.into(),
        ));
    }

    let rv = match rust_fn(local_args.as_slice()) {
        Ok(v) => v,
        Err(msg) => {
            ::set_last_error(&msg);
            return -1;
        }
    };
    let mut ret_val = *rv.value;
    let mut ret_type_code = rv.type_code as c_int;
    check_call!(ts::TVMCFuncSetReturn(
        ret,
        &mut ret_val as *mut _,
        &mut ret_type_code as *mut _,
        1 as c_int
    ));
    0
}

unsafe extern "C" fn tvm_callback_finalizer(fhandle: *mut c_void) {
    let rust_fn = unsafe {
        mem::transmute::<*mut c_void, fn(&[TVMArgValue]) -> Result<TVMRetValue<'static>>>(fhandle)
    };
    mem::drop(rust_fn);
}

fn convert_to_tvm_func(f: fn(&[TVMArgValue]) -> Result<TVMRetValue<'static>>) -> Function {
    let mut fhandle = ptr::null_mut() as ts::TVMFunctionHandle;
    let resource_handle = f as *mut fn(&[TVMArgValue]) -> Result<TVMRetValue<'static>>;
    check_call!(ts::TVMFuncCreateFromCFunc(
        Some(tvm_callback),
        resource_handle as *mut c_void,
        Some(tvm_callback_finalizer),
        &mut fhandle as *mut _
    ));
    Function::new(fhandle, false, false)
}

pub fn register(
    f: fn(&[TVMArgValue]) -> Result<TVMRetValue<'static>>,
    name: String,
    override_: bool,
) -> Result<()> {
    let func = convert_to_tvm_func(f);
    let ovd = if override_ { 1 } else { 0 };
    let name = CString::new(name).unwrap();
    check_call!(ts::TVMFuncRegisterGlobal(
        name.as_ptr() as *const c_char,
        func.as_handle(),
        ovd
    ));
    mem::forget(name);
    Ok(())
}

#[macro_export]
macro_rules! register_global_func {
    {
        $(#[$m:meta])*
        fn $fn_name:ident($args:ident : &[TVMArgValue]) -> Result<TVMRetValue> {
            $($code:tt)*
        }
    } => {{
        $(#[$m])*
        fn $fn_name($args: &[TVMArgValue]) -> Result<TVMRetValue<'static>> {
            $($code)*
        }

        $crate::function::register($fn_name, stringify!($fn_name).to_owned(), false).unwrap();
    }}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_global_func() {
        let list = list_global_func_names();
        assert!(list
            .lock()
            .unwrap()
            .iter()
            .find(|ref s| ***s == "tvm.graph_runtime.create")
            .is_some());
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
    fn provide_args() {
        let mut func = Builder::default();
        func.get_function("tvm.graph_runtime.remote_create".to_owned(), true, false)
            .args(&[10, 20])
            .arg(&"test".to_owned());
        assert!(func.arg_buf.is_some());
        assert_eq!(func.arg_buf.take().map(|bv| Vec::from(bv).len()), Some(3));
    }
}
