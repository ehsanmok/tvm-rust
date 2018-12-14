# TVM runtime frontend support

This crate provides idiomatic Rust API for [TVM](https://github.com/dmlc/tvm) runtime frontend.

Please follow the TVM [installation](https://docs.tvm.ai/install/index.html), with `export TVM_HOME=/path/to/tvm` and add `libtvm_runtime` to your `LD_LIBRARY_PATH`.

*Note:* To run the end-to-end examples and tests, `tvm`, `nnvm` and `topi` need to be added to your `PYTHONPATH`.

## Use TVM to Generate Shared Library

One can use the following Python snippet to generate `add_cpu.so` which add two vectors on CPU.

```python
import os
import tvm
from tvm.contrib import cc, util

def test_add(target_dir):
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    fadd = tvm.build(s, [A, B, C], "llvm", target_host="llvm", name="myadd")

    fadd.save(os.path.join(target_dir, "add_cpu.o"))
    cc.create_shared(os.path.join(target_dir, "add_cpu.so"),
            [os.path.join(target_dir, "add_cpu.o")])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])

```

## Run the Generated Shared Library

The following code snippet demonstrate how to load generated shared library (`add_cpu.so`).

```rust
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    let mut shape = vec![2];
    let mut data = vec![3f32, 4.0];

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
}
```
**Note:** it is required to instruct the `rustc` to link to the generated `add_cpu.so` in runtime, for example by
`cargo:rustc-link-search=native=add_cpu`. 

See the tests and examples custom `build.rs` for more details.

## Convert and Register a Rust Function as a TVM Packed Function

One can you the `register_global_func!` macro to convert and register a Rust's 
function of type `fn(&[TVMArgValue]) -> Result<TVMRetValue>` to a global TVM packed function as follows

```rust
#[macro_use]
extern crate tvm_frontend as tvm;

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
```