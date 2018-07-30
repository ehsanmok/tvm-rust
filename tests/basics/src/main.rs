extern crate ndarray;
extern crate tvm_rust as tvm;

fn main() {
    // let path = "./add_cpu";
//    let fadd = tvm::Module::load(&path, ".so");
//    let ctx = tvm::TVMContext::cpu(0);
//    let dtype = tvm::TVMType::from("float");
//    let mut shape = vec![1, 2];
//    let arr = tvm::ndarray::empty(&mut shape, ctx, dtype);
//    arr.copyfrom(&mut vec![3f64, 4f64]);
//    let ret = tvm::ndarray::empty(&mut shape, ctx, arr.dtype());
//    fadd::entry_func().unwrap().push_arg(arr).push_arg(arr).push_arg(ret);
//    fadd(());
//    assert_eq!(ret, vec![6f64, 8f64]);
}
