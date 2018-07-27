use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::cell::RefCell;
use std::collections::HashMap;

use tvm;

use function::Function;
use TVMResult;
use TVMTypeCode;
use TypeCode;
use TVMArgValue;
use TVMRetValue;
use TVMContext;

const ENTRY_FUNC: &'static str = "__tvm_main__";

thread_local! {
    static API: RefCell<HashMap<String, Function>> = RefCell::new(HashMap::new());
}

fn get(name: String) -> Option<Function> {
    API.with(|hm| {
        hm.borrow().get(&name).map(|f| f.clone())
    })
}

fn set(name: String, func: Function) {
    API.with(|hm| {
        (*hm.borrow_mut()).insert(name, func);
    })
}

fn get_api(name: String) -> Function {
    let mut func = get(name.clone());
    if func.is_none() {
        func = Function::get_function(format!("module.{}", name), false);
        set(name, func.clone().unwrap());
    }
    func.unwrap()
}

#[derive(Debug, Clone)]
pub struct Module {
    handle: tvm::TVMModuleHandle,
    is_global: bool,
    entry: Option<Function>,
}

impl Module {
    pub fn entry_func(&mut self) -> Option<Function> {
        if self.entry.is_none() {
            self.entry = Function::get_function(ENTRY_FUNC.to_owned(), false);
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
            Ok(Function::new(handle, false, None))
        }
    }

    pub fn import_module(&self, dependent_module: Module) {
        check_call!(tvm::TVMModImport(self.handle, dependent_module.handle))
    }

    pub fn load(path: &str, format: &str) -> TVMResult<()> {
        let mut func = get_api("_LoadFromFile".to_owned());
        // let file: &str = &format!("{}.{}", path, format);
        func.push_arg(path);
        func.push_arg(format);
        let ret = func.invoke().unwrap();
        assert_eq!(ret.type_code, TypeCode::kModuleHandle);
        Ok(ret.value)
        // unimplemented!()
    }

    // TODO: change to bool
    pub fn enabled(&self, target: &str) -> TVMResult<TVMRetValue> {
        let mut func = get_api("_Enabled".to_owned());
        func.push_arg(target);
        func.invoke()// .map(|ret| ret != 0)
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

impl TVMTypeCode for Module {
    fn type_code() -> TypeCode {
        TypeCode::kModuleHandle
    }
}

impl<'a> Drop for Module {
    fn drop(&mut self) {
        check_call!(tvm::TVMModFree(self.handle));
    }
}


#[cfg(test)]
mod tests {
    use super::*;

//    #[test]
//    fn enabled() {
//        let m = Module {
//            handle: ptr::null_mut() as *mut c_void,
//            is_global: false,
//            entry: None
//        };
//        println!("{:?}", m.enabled("cpu"));
//    }
}