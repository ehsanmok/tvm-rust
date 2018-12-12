#![feature(extern_crate_item_prelude, try_from)]
#![allow(unused_imports)]

#[macro_use]
extern crate tvm_frontend as tvm;

use std::convert::TryFrom;

extern crate ndarray as rust_ndarray;
use rust_ndarray::ArrayD;

use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
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
        .get_function("sum", true, false)
        .arg(&arr)
        .arg(&arr);
    assert_eq!(registered.invoke().unwrap().to_float(), 14f64);
}
