use std::cell::RefCell;
use std::collections::HashMap;
use std::mem;
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

use tvm;

use super::*;

const ENTRY_FUNC: &'static str = "__tvm_main__";

thread_local! {
 static API: RefCell<HashMap<String, Function>> = RefCell::new(HashMap::new());
}

fn get(name: String) -> Option<Function> {
 API.with(|hm| hm.borrow().get(&name).map(|f| f.clone()))
}

fn set(name: String, func: Function) {
 API.with(|hm| {
     (*hm.borrow_mut()).insert(name, func);
 })
}

fn get_api(name: String) -> Function {
 let mut func = get(name.clone());
 if func.is_none() {
     func = Function::get_function(format!("{}", name), true, false);
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
         Ok(Function::new(handle, false, false))
     }
 }

 pub fn import_module(&self, dependent_module: Module) {
     check_call!(tvm::TVMModImport(self.handle, dependent_module.handle))
 }

 pub fn load(path: &str, format: &str) -> TVMResult<()> {
     let func = get_api("module._LoadFromFile".to_owned());
     //println!("{:?}", func);
     let ret = function::Builder::from(func)
          .push_arg(path)
          .push_arg(format);
     //println!("path pointer: {:?}", path.as_ptr());
     //println!("{:?}", ret);
     println!("result: {:?}", ret.invoke().unwrap());
//      assert_eq!(ret.type_code, TypeCode::kModuleHandle);
      Ok(())
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

impl Drop for Module {
 fn drop(&mut self) {
     if !self.is_global {
         check_call!(tvm::TVMModFree(self.handle));
     }
 }
}

