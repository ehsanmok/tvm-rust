# TVM runtime frontend support (pre-release)

This crate provides idiomatic Rust API for [TVM](https://github.com/dmlc/tvm) runtime frontend as part of [RFC 1601](https://github.com/dmlc/tvm/issues/1601). Currently this requires **Nightly Rust**.

Checkout the [docs](https://ehsanmok.github.io/tvm_frontend/tvm_frontend/index.html)

## What does this crate offer?

Here is a major workflow

1. Train your Deep Learning model using any major deep learning framework [PyTorch](https://pytorch.org/), [Apache MXNet](https://mxnet.incubator.apache.org/) and [TensorFlow](https://www.tensorflow.org/)
2. Use TVM to build optimized model artifacts on a given supported TVM context such as CPU, GPU, OpenCL, Vulkan, VPI, ROCM, etc.
3. Deploy your models using Rust :heart:

### Example: Resnet18 pretrained on Imagenet

Please checkout [examples/resnet](https://github.com/ehsanmok/tvm-rust/tree/master/examples/resnet) for the complete end-to-end example.

Here's python snippet for download and building Resnet18 via MXNet and TVM

```python
block = get_model('resnet18_v1', pretrained=True)
    
sym, params = nnvm.frontend.from_mxnet(block)
# add the softmax layer for prediction
net = nnvm.sym.softmax(sym)
# compile the model
with nnvm.compiler.build_config(opt_level=opt_level):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape={"data": data_shape}, params=params)
# same the model artifacts
lib.save(os.path.join(target_dir, "deploy_lib.o"))
cc.create_shared(os.path.join(target_dir, "deploy_lib.so"),
                [os.path.join(target_dir, "deploy_lib.o")])

with open(os.path.join(target_dir, "deploy_graph.json"), "w") as fo:
    fo.write(graph.json())
with open(os.path.join(target_dir,"deploy_param.params"), "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
```

Now, we can read input the artifacts to create and run the Graph Runtime to detect our cat image

![cat](https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true)

as demostrated in the following Rust snippet

```rust

let graph = fs::read_to_string("deploy_graph.json")?;
// load module
let lib = Module::load(&Path::new("deploy_lib.so"))?;
// get the global TVM graph runtime function
let runtime_create_fn = Function::get_function("tvm.graph_runtime.create", true).unwrap();

let runtime_create_fn_ret = call_packed!(
    runtime_create_fn,
    &graph,
    &lib,
    &ctx.device_type,
    &ctx.device_id
)?;
// get graph runtime module
let graph_runtime_module = runtime_create_fn_ret.to_module();
// get the registered `load_params` from runtime module
let load_param_fn = graph_runtime_module
    .get_function("load_params", false)
    .unwrap();
// parse parameters and convert to TVMByteArray
let params: Vec<u8> = fs::read("deploy_param.params")?;
let barr = TVMByteArray::from(&params);
// load the parameters
call_packed!(load_param_fn, &barr)?;
// get the set_input function
let set_input_fn = graph_runtime_module
    .get_function("set_input", false)
    .unwrap();

call_packed!(set_input_fn, "data", &input)?;
// get `run` function from runtime module
let run_fn = graph_runtime_module.get_function("run", false).unwrap();
// execute the run function. Note that it has no argument.
call_packed!(run_fn,)?;
// prepare to get the output
let output_shape = &mut [1, 1000];
let output = empty(output_shape, TVMContext::cpu(0), TVMType::from("float"));
// get the `get_output` function from runtime module
let get_output_fn = graph_runtime_module
    .get_function("get_output", false)
    .unwrap();
// execute the get output function
call_packed!(get_output_fn, &0, &output)?;
// flatten the output as Vec<f32>
let output = output.to_vec::<f32>()?;
```

## Installation

Please follow the TVM [installation](https://docs.tvm.ai/install/index.html), `export TVM_HOME=/path/to/tvm` and add `libtvm_runtime` to your `LD_LIBRARY_PATH`.

*Note:* To run the end-to-end examples and tests, `tvm`, `nnvm` and `topi` need to be added to your `PYTHONPATH`.

## Other supported functionalities

### Use TVM to Generate Shared Library

One can use the following Python snippet to generate `add_gpu.so` which add two vectors on GPU.

```python
import os

import tvm
from tvm.contrib import cc


def test_add(target_dir):
    if not tvm.module.enabled("cuda"):
        print(f"skip {__file__} because cuda is not enabled...")
        return
    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")

    s = tvm.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    fadd_cuda = tvm.build(s, [A, B, C], "cuda", target_host="llvm", name="myadd")

    fadd_cuda.save(os.path.join(target_dir, "add_gpu.o"))
    fadd_cuda.imported_modules[0].save(os.path.join(target_dir, "add_gpu.ptx"))
    cc.create_shared(os.path.join(target_dir, "add_gpu.so"),
            [os.path.join(target_dir, "add_gpu.o")])


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(-1)
    test_add(sys.argv[1])
```

### Run the Generated Shared Library

The following code snippet demonstrate how to load generated shared library (`add_cpu.so`).

```rust
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = empty(shape, TVMContext::gpu(0), TVMType::from("float"));
    arr.copy_from_buffer(data.as_mut_slice());
    let mut ret = empty(shape, TVMContext::gpu(0), TVMType::from("float"));
    let path = Path::new("add_gpu.so");
    let ptx = Path::new("add_gpu.ptx");
    let mut fadd = Module::load(path).unwrap();
    let fadd_dep = Module::load(ptx).unwrap();
    assert!(fadd.enabled("gpu"));
    fadd.import_module(fadd_dep);
    fadd.entry_func();
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

### Convert and Register a Rust Function as a TVM Packed Function

One can you the `register_global_func!` macro to convert and register a Rust's 
function of type `fn(&[TVMArgValue]) -> Result<TVMRetValue>` to a *global TVM packed function* as follows

```rust
#[macro_use]
extern crate tvm_frontend as tvm;

use tvm::*;

fn main() {
    register_global_func! {
        fn sum(args: &[TVMArgValue]) -> Result<TVMRetValue> {
            let mut ret = 0f32;
            let shape = &mut [2];
            for arg in args.iter() {
                let e = empty(shape, TVMContext::cpu(0), TVMType::from("float"));
                let arr = arg.to_ndarray().copy_to_ndarray(e).unwrap();
                let rnd: ArrayD<f32> = ArrayD::try_from(&arr).unwrap();
                ret += rnd.scalar_sum();
            }
            let ret_val = TVMRetValue::from(&ret);
            Ok(ret_val)
        }
    }

    let shape = &mut [2];
    let mut data = vec![3f32, 4.0];
    let mut arr = empty(shape, TVMContext::cpu(0), TVMType::from("float"));
    arr.copy_from_buffer(data.as_mut_slice());

    let mut registered = function::Builder::default();
    registered
        .get_function("sum", true)
        .arg(&arr)
        .arg(&arr);
    assert_eq!(registered.invoke().unwrap().to_float(), 14f64);
    }
```
