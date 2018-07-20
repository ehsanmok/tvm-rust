use std::mem;
use std::os::raw::{c_char, c_int};
use std::ptr;

use tvm;

use function::Function;
use TVMResult;
use TVMTypeCode;
use TypeCode;

const ENTRY_FUNC: &'static str = "__tvm_main__";

#[derive(Debug)]
pub struct Module {
    handle: tvm::TVMModuleHandle,
    is_global: bool,
    entry: Option<Function>,
}

impl Module {
    pub fn entry_func(&mut self) -> Option<Function> {
        if self.entry.is_none() {
            self.entry = Function::get_function(ENTRY_FUNC, false);
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
            Ok(Function::new(handle, false))
        }
    }

    pub fn import_module(&self, dependent_module: Module) {
        check_call!(tvm::TVMModImport(self.handle, dependent_module.handle))
    }

    pub fn load(file_name: &str, format: &str) -> Module {
        unimplemented!()
    }

    pub fn enabled(target: &str) -> bool {
        unimplemented!()
    }

    pub fn as_handle(&self) -> tvm::TVMModuleHandle {
        self.handle
    }

    pub fn is_global(&self) -> bool {
        self.is_global
    }
}

impl TVMTypeCode for Module {
    fn type_code() -> TypeCode {
        TypeCode::kModuleHandle
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        check_call!(tvm::TVMModFree(self.handle));
    }
}
