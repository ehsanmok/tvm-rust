#![feature(extern_crate_item_prelude)]

#[allow(unused_imports)]
#[macro_use]
extern crate tvm_rust as tvm;

use tvm::*;

fn main() {
    // fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue<'static>> {
    //     let mut ret = 0;
    //     for arg in args.iter() {
    //         ret += arg.to_int();
    //     }
    //     let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
    //     Ok(ret_val)
    // }
    // tvm::function::register(sum, "sum".to_owned(), false).unwrap();

    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
            let mut ret = 0;
            for arg in args.iter() {
                ret += arg.to_int();
            }
            let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
            Ok(ret_val)
        }
    }

    let mut registered = function::Builder::default().get_function("sum".to_owned(), true, false);
    assert!(registered.func.is_some());
    registered = registered.push_arg(&10).push_arg(&20);
    assert_eq!(registered.invoke().unwrap().to_int(), 30);

    // must be run separately!
    // fails with unmatched type_code
    // register_global_func! {
    // 	fn concate_str(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
    // 	    let mut ret = "".to_owned();
    // 	    for arg in args.iter() {
    // 	        ret += arg.to_str();
    // 	    }
    // 	    let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
    // 	    Ok(ret_val)
    // 	}
    // }
    // let mut registered = function::Builder::default().get_function("concate_str".to_owned(), true, false);
    //    assert!(registered.func.is_some());
    //    registered = registered.push_arg("a").push_arg("b");
    //    assert_eq!(registered.invoke().unwrap().to_str(), "ab");
}
