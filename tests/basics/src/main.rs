extern crate ndarray as rust_ndarray;
extern crate tvm_rust as tvm;

use std::path::Path;
use tvm::*;

fn main() {
    let mut shape = vec![2];
    let mut data = vec![3f32, 4.0];

    if cfg!(features="default") {
        let mut arr = tvm::ndarray::empty(
            &mut shape,
            TVMContext::cpu(0),
            TVMType::from("float"),
        );

        arr.copy_from(&mut data);

        let mut ret = tvm::ndarray::empty(
            &mut shape,
            TVMContext::cpu(0),
            TVMType::from("float"),
        );
        let path = Path::new("add_cpu.so");
        let mut fadd = tvm::Module::load(path).unwrap();
        fadd = fadd.entry_func();
        tvm::function::Builder::from(&mut fadd)
            .push_arg(&arr)
            .push_arg(&arr)
            .accept_ret(&mut ret)
            .invoke()
            .unwrap();

        assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
    }

    if cfg!(features="gpu") {
        let mut arr = tvm::ndarray::empty(
            &mut shape,
            TVMContext::gpu(0),
            TVMType::from("float"),
        );

        arr.copy_from(&mut data);

        let mut ret = tvm::ndarray::empty(
            &mut shape,
            TVMContext::gpu(0),
            TVMType::from("float"),
        );
        let path = Path::new("add_gpu.so");
        let ptx = Path::new("add_gpu.ptx");
        let mut fadd = tvm::Module::load(path).unwrap();
        let fadd_dep = tvm::Module::load(ptx).unwrap();
        fadd.import_module(fadd_dep);
        fadd = fadd.entry_func();
        tvm::function::Builder::from(&mut fadd)
            .push_arg(&arr)
            .push_arg(&arr)
            .accept_ret(&mut ret)
            .invoke()
            .unwrap();

        assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
    }

}
