use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use tvm;

use super::*;

const ENTRY_FUNC: &'static str = "__tvm_main__";

#[derive(Debug, Clone)]
pub struct Module<'a> {
    handle: tvm::TVMModuleHandle,
    is_global: bool,
    entry: Option<Function<'a>>,
}

impl<'a> Module<'a> {
    pub fn entry_func(&mut self) -> Option<Function> {
        if self.entry.is_none() {
            self.entry = Function::get_function(ENTRY_FUNC.to_owned(), false, false);
        }
        self.entry.take()
    }

    pub fn get_function(name: &str, query_import: bool) -> TVMResult<Function> {
        let query_import = if query_import == true { 1 } else { 0 };
        let mut handle = ptr::null_mut() as tvm::TVMModuleHandle;
        check_call!(tvm::TVMModGetFunction(
            handle,
            name.as_ptr() as *const c_char,
            query_import as c_int,
            &mut handle as *mut _
        ));
        if handle.is_null() {
            panic!("Module has no function {}", name);
        } else {
            Ok(Function::new(Box::new(handle), false, false, None))
        }
    }

    pub fn import_module(&self, dependent_module: Module) {
        check_call!(tvm::TVMModImport(self.handle, dependent_module.handle))
    }

    pub fn load(path: &str, format: &str) -> TVMResult<()> {
        // let mut func = get_api("_LoadFromFile".to_owned());
        // func.push_arg(path);
        // func.push_arg(format);
        // let ret = func.invoke();
        // assert_eq!(ret.type_code, TypeCode::kModuleHandle);
        // Ok(())
        unimplemented!()
    }

    pub fn enabled(&self, target: &str) -> bool {
        // let mut func = get_api("_Enabled".to_owned());
        // func.push_arg(target);
        // func.invoke().value != TVMValue::default()
        unimplemented!()
    }

    pub fn as_handle(&self) -> tvm::TVMModuleHandle {
        self.handle
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }

    pub fn as_module(&self) -> Self {
        self.clone()
    }
}

// impl TVMTypeCode for Module {
//     fn type_code() -> TypeCode {
//         TypeCode::kModuleHandle
//     }
// }

impl<'a> Drop for Module<'a> {
    fn drop(&mut self) {
        check_call!(tvm::TVMModFree(self.handle));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn enabled() {
    //     let m = Module {
    //         handle: ptr::null_mut() as *mut c_void,
    //         is_global: false,
    //         entry: None,
    //     };
    //     println!("{:?}", m.enabled("cpu"));
    // }
}
