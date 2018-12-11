#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_rust as tvm;

use tvm::*;

fn main() {
    register_global_func! {
        fn concate_str(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = "".to_owned();
            for arg in args.iter() {
                ret += arg.to_string().as_str();
            }
            let ret_val = TVMRetValue::from(&ret);
            Ok(ret_val)
        }
    }
    let mut registered = function::Builder::default();
    registered.get_function("concate_str", true, false);
    assert!(registered.func.is_some());
    let a = "a".to_string();
    let b = "b".to_string();
    let c = "c".to_string();
    registered.arg(&a).arg(&b).arg(&c);
    assert_eq!(registered.invoke().unwrap().to_string(), "abc".to_owned());
}
