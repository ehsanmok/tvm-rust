use std::mem;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;

use tvm;

use function::{self, Function};
use internal_api;
use TVMResult;

const ENTRY_FUNC: &'static str = "__tvm_main__";

#[derive(Debug, Clone)]
pub struct Module {
    handle: tvm::TVMModuleHandle,
    is_released: bool,
    pub(crate) entry: Option<Function>,
}

impl Module {
    fn new(handle: tvm::TVMModuleHandle, is_released: bool, entry: Option<Function>) -> Self {
        Self {
            handle,
            is_released,
            entry,
        }
    }

    pub fn entry_func(mut self) -> Self {
        if self.entry.is_none() {
            self.entry = self.get_function(ENTRY_FUNC.to_owned(), false).ok();
        }
        self
    }

    pub fn get_function(&mut self, name: String, query_import: bool) -> TVMResult<Function> {
        let name = name.to_owned();
        let query_import = if query_import == true { 1 } else { 0 };
        let mut fhandle = ptr::null_mut() as tvm::TVMFunctionHandle;
        check_call!(tvm::TVMModGetFunction(
            self.handle,
            name.as_ptr() as *const c_char,
            query_import as c_int,
            &mut fhandle as *mut _
        ));
        if fhandle.is_null() {
            panic!("function handle is null for {}", name);
        } else {
            mem::forget(name);
            Ok(Function::new(fhandle, false, false))
        }
    }

    pub fn import_module(&self, dependent_module: Module) {
        check_call!(tvm::TVMModImport(self.handle, dependent_module.handle))
    }

    pub fn load(path: &Path) -> TVMResult<Module> {
        let mut module_handle = ptr::null_mut() as tvm::TVMModuleHandle;
        let path = path.to_owned();
        check_call!(tvm::TVMModLoadFromFile(
            path.to_str().unwrap().as_ptr() as *const c_char,
            path.extension().unwrap().to_str().unwrap().as_ptr() as *const c_char,
            &mut module_handle as *mut _
        ));
        Ok(Self::new(module_handle, false, None))
    }

    pub fn enabled(&self, target: &str) -> bool {
        let target = target.to_owned();
        let func = internal_api::get_api("module._Enabled".to_owned());
        let ret = function::Builder::from(func)
            .push_arg(&target)
            .invoke()
            .unwrap();
        mem::forget(target);
        unsafe { ret.value.v_int64 != 0 }
    }

    pub fn as_handle(&self) -> tvm::TVMModuleHandle {
        self.handle
    }

    pub fn is_released(&self) -> bool {
        self.is_released
    }

    pub fn as_module(&self) -> Self {
        self.clone()
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.is_released {
            check_call!(tvm::TVMModFree(self.handle));
            self.is_released = true;
        }
    }
}
