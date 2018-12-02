#![feature(extern_crate_item_prelude)]
#![allow(unused_imports)]

extern crate tvm_rust as tvm;

use tvm::*;

fn main() {
    fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue<'static>> {
        let mut ret = 0;
        for arg in args.iter() {
            ret += arg.to_int();
        }
        let ret_val = TVMRetValue::from(&ret);
        Ok(ret_val)
    }
    tvm::function::register(sum, "mysum".to_owned(), false).unwrap();

    let mut registered = function::Builder::default();
    registered.get_function("mysum".to_owned(), true, false);
    assert!(registered.func.is_some());
    registered.args(&[10, 20, 30]);
    assert_eq!(registered.invoke().unwrap().to_int(), 60);
}
