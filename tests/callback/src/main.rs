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
    // fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue<'static>> {
    //     let mut ret = 0;
    //     for arg in args.iter() {
    //         ret += arg.to_int();
    //     }
    //     let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
    //     Ok(ret_val)
    // }
    // tvm::function::register(sum, "sum".to_owned(), false).unwrap();

    // register_global_func! {
    //     fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
    //         let mut ret = 0f64;
    //         for arg in args.iter() {
    //             ret += arg.to_float();
    //         }
    //         let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
    //         Ok(ret_val)
    //     }
    // }

    // let mut registered = function::Builder::default();
    // registered.get_function("sum".to_owned(), true, false);
    // assert!(registered.func.is_some());
    // registered.arg(&10f64).arg(&20f64);
    // assert_eq!(registered.invoke().unwrap().to_float(), 30f64);

    // must be run separately!
    // String is hard!
    // register_global_func! {
    // 	fn concate_str(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
    // 	    let mut ret = "".to_owned();
    // 	    for arg in args.iter() {
    //             // ret = format!("{}{}", ret, arg.to_string());
    //             ret += &arg.to_string().trim();
    // 	    }
    // 	    let ret_val = TVMRetValue::new(TVMValue::from(&ret), TypeCode::from(&ret));
    //         mem::forget(ret);
    // 	    Ok(ret_val)
    // 	}
    // }
    // let mut registered = function::Builder::default();
    // registered.get_function("concate_str".to_owned(), true, false);
    // assert!(registered.func.is_some());
    // registered.arg(&"a".to_owned()).arg(&"b".to_owned()).arg(&"c".to_owned());
    // println!("{:?}", registered.invoke().unwrap().to_string());
    // assert_eq!(registered.invoke().unwrap().to_string(), "ab");

    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> TVMResult<TVMRetValue> {
            let mut ret = 0f32;
            let mut shape = vec![2];
            for arg in args.iter() {
                let e = empty(&mut shape, TVMContext::cpu(0), TVMType::from("float"));
                let arr = arg.to_ndarray().copy_to_ndarray(e).unwrap();
                let rnd: ArrayD<f32> = ArrayD::try_from(&arr).unwrap();
                ret += rnd.scalar_sum();
            }
            let ret_val = TVMRetValue::from(&ret);
            Ok(ret_val)
        }
    }

    let mut shape = vec![2];
    let mut data = vec![3f32, 4.0];
    let mut arr = empty(&mut shape, TVMContext::cpu(0), TVMType::from("float"));
    arr.copy_from_buffer(data.as_mut_slice());

    let mut registered = function::Builder::default();
    registered
        .get_function("sum".to_owned(), true, false)
        .arg(&arr)
        .arg(&arr);
    assert_eq!(registered.invoke().unwrap().to_float(), 14f64);
}
