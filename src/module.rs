use std::mem;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;

use ts;

use function::{self, Function};
use internal_api;
use TVMError;
use TVMResult;

const ENTRY_FUNC: &'static str = "__tvm_main__";

#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) handle: ts::TVMModuleHandle,
    is_released: bool,
    pub(crate) entry: Option<Function>,
}

impl Module {
    pub(crate) fn new(
        handle: ts::TVMModuleHandle,
        is_released: bool,
        entry: Option<Function>,
    ) -> Self {
        Self {
            handle,
            is_released,
            entry,
        }
    }

    pub fn entry_func(mut self) -> Self {
        if self.entry.is_none() {
            self.entry = self.get_function(ENTRY_FUNC, false).ok();
        }
        self
    }

    pub fn get_function(&self, name: &str, query_import: bool) -> TVMResult<Function> {
        let name = name.to_owned();
        let query_import = if query_import == true { 1 } else { 0 };
        let mut fhandle = ptr::null_mut() as ts::TVMFunctionHandle;
        check_call!(ts::TVMModGetFunction(
            self.handle,
            name.as_ptr() as *const c_char,
            query_import as c_int,
            &mut fhandle as *mut _
        ));
        if fhandle.is_null() {
            return Err(TVMError::new(stringify!(
                "function handle is null for {}",
                name
            )));
        } else {
            mem::forget(name);
            Ok(Function::new(fhandle, false, false))
        }
    }

    pub fn import_module(&self, dependent_module: Module) {
        check_call!(ts::TVMModImport(self.handle, dependent_module.handle))
    }

    pub fn load(path: &Path) -> TVMResult<Module> {
        let mut module_handle = ptr::null_mut() as ts::TVMModuleHandle;
        let path = path.to_owned();
        check_call!(ts::TVMModLoadFromFile(
            path.to_str().unwrap().as_ptr() as *const c_char,
            path.extension().unwrap().to_str().unwrap().as_ptr() as *const c_char,
            &mut module_handle as *mut _
        ));
        let ret = Self::new(module_handle, false, None);
        // Ok(ret.entry_func())
        Ok(ret)
    }

    pub fn enabled(&self, target: &str) -> bool {
        let target = target.to_owned();
        let func = internal_api::get_api("module._Enabled".to_owned());
        let ret = function::Builder::from(func)
            .push_arg(&target)
            .invoke()
            .unwrap();
        mem::forget(target);
        ret.to_int() != 0
    }

    pub fn as_handle(&self) -> ts::TVMModuleHandle {
        self.handle
    }

    pub fn is_released(&self) -> bool {
        self.is_released
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.is_released {
            check_call!(ts::TVMModFree(self.handle));
            self.is_released = true;
        }
    }
}
