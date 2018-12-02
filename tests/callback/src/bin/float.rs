#![feature(extern_crate_item_prelude)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_rust as tvm;

use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = 0f64;
            for arg in args.iter() {
                ret += arg.to_float();
            }
            let ret_val = TVMRetValue::from(&ret);
            Ok(ret_val)
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("sum".to_owned(), true, false);
    assert!(registered.func.is_some());
    registered.args(&[10f64, 20f64, 30f64]);
    assert_eq!(registered.invoke().unwrap().to_float(), 60f64);
}
