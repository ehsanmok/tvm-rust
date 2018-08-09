#![allow(unused_variables, unused_mut)]

extern crate ndarray;
extern crate tvm_rust as tvm;

use std::path::Path;

fn main() {
    let path = Path::new("add_cpu.so");
    let mut fadd = tvm::Module::load(path).unwrap();
    let ctx = tvm::TVMContext::cpu(0);
    let dtype = tvm::TVMType::from("float");
    let mut shape = vec![1, 2];
    let mut arr = tvm::ndarray::empty(&mut shape, ctx.clone(), dtype);
    arr.copyfrom(&mut vec![3f32, 4.]);
    let mut ret = tvm::ndarray::empty(&mut shape, ctx, arr.dtype());
    fadd = fadd.entry_func();
    let f = tvm::function::Builder::from(&mut fadd)
        .push_arg(&arr)
        .push_arg(&arr)
        .accept_ret(&mut ret).invoke().unwrap();
    println!("{:?}", f);
//        .invoke()
//        .unwrap();
    //println!("{:?}", ret);
//    assert_eq!(ret, vec![6f64, 8f64]);
}
