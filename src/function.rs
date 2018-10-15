use std::convert::TryFrom;
use std::ffi::CStr;
use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::slice;
use std::str;
use std::sync::Mutex;

use tvm;

use Module;
use TVMArgValue;
use TVMError;
use TVMResult;
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
    check_call!(tvm::TVMFuncListGlobalNames(
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

pub fn get_global_func(
    name: &'static str,
    is_global: bool,
    allow_missing: bool,
) -> Option<Function> {
    let name = name.to_owned();
    let mut handle = ptr::null_mut() as tvm::TVMFunctionHandle;
    check_call!(tvm::TVMFuncGetGlobal(
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
            panic!("Cannot find global function {}", name);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Builder<'a> {
    func: Option<Function>,
    arg_buf: Option<Box<[TVMArgValue<'a>]>>,
    ret_buf: Option<Box<[TVMRetValue<'a>]>>,
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

    pub fn get_function(mut self, name: String, is_global: bool, allow_missing: bool) -> Self {
        self.func = Function::get_function(name, is_global, allow_missing);
        self
    }

    pub fn push_arg<'b, T: 'b + ?Sized>(mut self, arg: &'b T) -> Self
    where
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        let tvm_arg = TVMArgValue::new(TVMValue::from(arg), TypeCode::from(arg));
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
                new_buf.into_boxed_slice()
            });
            Self::new(self.func, new_arg_buf, self.ret_buf)
        }
    }

    pub fn accept_ret<'b, T: 'b + ?Sized>(mut self, arg: &'b mut T) -> Self
    where
        TVMValue: From<&'b T>,
        TypeCode: From<&'b T>,
    {
        let tvm_ret = TVMRetValue::new(TVMValue::from(arg), TypeCode::from(arg));
        if self.ret_buf.is_none() {
            self.ret_buf = Some(Box::new([tvm_ret]));
            return self;
        } else {
            let new_ret_buf = self.ret_buf.take().map(|bbuf| {
                let mut new_buf = Vec::with_capacity(1);
                new_buf.push(tvm_ret);
                new_buf.into_boxed_slice()
            });
            Self::new(self.func, self.arg_buf, new_ret_buf)
        }
    }

    pub fn invoke(self) -> TVMResult<TVMRetValue<'a>> {
        self(())
    }
}

impl<'a> FnOnce<((),)> for Builder<'a> {
    type Output = TVMResult<TVMRetValue<'a>>;
    extern "rust-call" fn call_once(self, _: ((),)) -> Self::Output {
        if self.func.is_none() {
            panic!("Function handle is None");
        }
        let mut ret_val = tvm::TVMValue { v_int64: 0 };
        let mut ret_type_code = 0;
        let arg_buf = self.arg_buf.clone().unwrap();
        let mut num_args = arg_buf.len();
        let mut values = arg_buf
            .iter()
            .map(|tav| tav.clone().value.inner)
            .collect::<Vec<tvm::TVMValue>>();
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
        check_call!(tvm::TVMFuncCall(
            self.func.unwrap().handle,
            values.as_mut_ptr(),
            tcodes.as_mut_ptr(),
            num_args as c_int,
            &mut ret_val as *mut _,
            &mut ret_type_code as *mut _
        ));
        let ret = TVMRetValue::new(
            TVMValue::new(ValueKind::Return, ret_val),
            TypeCode::from(&ret_type_code),
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

#[derive(Debug, Clone, Hash)]
pub struct Function {
    handle: tvm::TVMFunctionHandle,
    is_global: bool,
    is_released: bool,
}

impl Function {
    pub fn new(handle: tvm::TVMFunctionHandle, is_global: bool, is_released: bool) -> Self {
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

    pub fn as_handle(&self) -> tvm::TVMFunctionHandle {
        self.handle
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }

    pub fn is_released(&self) -> bool {
        self.is_released
    }
}

// macro_rules! tvm_callback {
//     { $fn_name:ident, $cb:item } => (
//         unsafe extern "C" fn $fn_name(
//             args: *mut tvm::TVMValue,
//             type_codes: *mut c_int,
//             num_args: c_int,
//             ret: tvm::TVMRetValueHandle,
//             resource_handle: *mut c_void,
//         ) -> c_int {
//             let mut value = tvm::TVMValue { v_int64: 0 };
//             let mut tcode = 0;
//             let mut rargs = vec![];
//             for arg in 0..num_args {
//                 // handle TVMCbArgToReturn
//                 rargs.push(args);
//             }

//             let local_rfn = unsafe { *resource_handle };
//             let rv = local_rfn(rargs); // --> so local_rfn must return Result<TVMRetValue>
//             // handle rv error! --> TVMAPISetLastError --> return -1
//             match rv {
//                 Ok(_) => { ... },
//                 Err(msg) => TVMError::set_last(msg),
//             }

//             check_call!(tvm::TVMCFuncSetReturn(ret,
//                                                 &mut value as *mut _,
//                                                 &mut tcode as *mut _,
//                                                 1))

//             // how to wrap the ret idomatically?
//             // TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret))
//             0
//         }
//     )
// }

macro_rules! register_func {
    {
        $(#[$m:meta])*
        fn $fn_name:ident($arg_name:ident : $arg_ty:ty) -> $ret:ty {
            $($code:tt)*
        }
    } => {
        $(#[$m])*
        unsafe extern "C" fn $fn_name(args: $arg_ty.inner as *tvm::TVMValue)

    }
}

#[derive(Debug, Clone)]
pub struct BackendPackedCFunc {
    pub func: tvm::TVMPackedCFunc,
    pub finalizer: tvm::TVMPackedCFuncFinalizer,
}

impl BackendPackedCFunc {
    pub fn new(func: tvm::TVMPackedCFunc, finalizer: tvm::TVMPackedCFuncFinalizer) -> Self {
        Self { func, finalizer }
    }
}

/// Convert `BackendPackedCFunc` to `Function`
impl<'a> TryFrom<&'a mut BackendPackedCFunc> for Function {
    type Error = TVMError;
    fn try_from(packed: &mut BackendPackedCFunc) -> TVMResult<Self> {
        packed
            .func
            .take()
            .map(|func| {
                let mut fhandle = ptr::null_mut() as tvm::TVMFunctionHandle;
                let resource_handle = ptr::null_mut() as *mut c_void;
                check_call!(tvm::TVMFuncCreateFromCFunc(
                    Some(func),
                    resource_handle,
                    packed.finalizer,
                    &mut fhandle as *mut _
                ));
                Function::new(fhandle, false, false)
            }).ok_or(TVMError::new("Packed function is not provided"))
    }
}

impl Drop for Function {
    fn drop(&mut self) {
        if !self.is_released {
            if !self.is_global {
                check_call!(tvm::TVMFuncFree(self.handle));
                self.is_released = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_global_func() {
        let list = list_global_func_names();
        assert!(
            list.lock()
                .unwrap()
                .iter()
                .find(|ref s| ***s == "tvm.graph_runtime.create")
                .is_some()
        );
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
        let mut func = Builder::default().get_function(
            "tvm.graph_runtime.remote_create".to_owned(),
            true,
            false,
        );
        func = func.push_arg(&10).push_arg("test");
        assert!(func.arg_buf.is_some());
        assert_eq!(func.arg_buf.take().map(|bv| Vec::from(bv).len()), Some(2));
    }

    fn reg_func() {
        register_func!{ 
            False,
        fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue<'static>> {
            let mut ret = 0;
            for arg in args.iter() {
                ret += arg.to_int();
            }

            let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
            Ok(ret_val)
        }};
    }
}
