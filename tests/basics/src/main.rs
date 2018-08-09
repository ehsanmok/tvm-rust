extern crate ndarray;
extern crate tvm_rust as tvm;

use std::path::Path;

fn main() {
    let path = Path::new("add_cpu.so");
    let mut fadd = tvm::Module::load(path).unwrap();
    let mut shape = vec![1, 2];
    let mut data = vec![3f32, 4.0];
    let mut arr = tvm::ndarray::empty(
        &mut shape,
        tvm::TVMContext::cpu(0),
        tvm::TVMType::from("float"),
    );
    arr.copyfrom(&mut data);
    //println!("{:?}", arr.to_vec::<f32>().unwrap());
    let mut ret = tvm::ndarray::empty(
        &mut shape,
        tvm::TVMContext::cpu(0),
        tvm::TVMType::from("float"),
    );
    fadd = fadd.entry_func();
    let f = tvm::function::Builder::from(&mut fadd)
        .push_arg(&arr)
        .push_arg(&arr)
        .accept_ret(&mut ret)
        .invoke()
        .unwrap();
    println!("{:?}", f);
    // assert_eq!(ret, vec![6f64, 8f64]);
}
