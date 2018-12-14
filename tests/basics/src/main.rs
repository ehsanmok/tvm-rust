extern crate ndarray as rust_ndarray;
extern crate tvm_frontend as tvm;

use std::path::Path;

use tvm::*;

fn main() {
    println!("start integration test");
    let mut shape = vec![2];
    let mut data = vec![3f32, 4.0];

    if cfg!(feature = "cpu") {
        println!("cpu test");
        let mut arr = empty(&mut shape, TVMContext::cpu(0), TVMType::from("float"));

        arr.copy_from_buffer(data.as_mut_slice());

        let mut ret = empty(&mut shape, TVMContext::cpu(0), TVMType::from("float"));
        let path = Path::new("add_cpu.so");
        let mut fadd = Module::load(&path).unwrap();
        assert!(fadd.enabled("cpu".to_owned()));
        fadd = fadd.entry_func();
        function::Builder::from(&mut fadd)
            .arg(&arr)
            .arg(&arr)
            .set_output(&mut ret)
            .invoke()
            .unwrap();

        assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
        println!("success!")
    }

    if cfg!(feature = "gpu") {
        println!("gpu test");
        let mut arr = empty(&mut shape, TVMContext::gpu(0), TVMType::from("float"));

        arr.copy_from_buffer(data.as_mut_slice());

        let mut ret = empty(&mut shape, TVMContext::gpu(0), TVMType::from("float"));
        let path = Path::new("add_gpu.so");
        let ptx = Path::new("add_gpu.ptx");
        let mut fadd = Module::load(path).unwrap();
        let fadd_dep = Module::load(ptx).unwrap();
        assert!(fadd.enabled("gpu".to_owned()), "GPU is not enabled!");
        fadd.import_module(fadd_dep);
        fadd = fadd.entry_func();
        function::Builder::from(&mut fadd)
            .arg(&arr)
            .arg(&arr)
            .set_output(&mut ret)
            .invoke()
            .unwrap();

        assert_eq!(ret.to_vec::<f32>().unwrap(), vec![6f32, 8.0]);
        println!("success!")
    }
}
