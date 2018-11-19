#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_rust as tvm;

use std::convert::TryFrom;

extern crate ndarray as rust_ndarray;
use rust_ndarray::ArrayD;

use std::mem;

use tvm::*;

fn main() {
    // FIXME: apparently mem::forget is not enough
    register_global_func! {
    	fn concate_str(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
    	    let mut ret = "".to_owned();
    	    for arg in args.iter() {
                ret.push_str(arg.to_string().trim());
                println!("ret {:?}", ret);
    	    }
    	    let ret_val = TVMRetValue::from(&ret);
            mem::forget(ret);
    	    Ok(ret_val)
    	}
    }
    let mut registered = function::Builder::default();
    registered.get_function("concate_str".to_owned(), true, false);
    assert!(registered.func.is_some());
    let a = "a".to_string();
    let b = "b".to_string();
    registered.arg(&a).arg(&b);
    println!("{:?}", registered.invoke().unwrap().to_string());
    //assert_eq!(registered.invoke().unwrap().to_string(), "ab".to_owned());
}
