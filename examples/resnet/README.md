## Resnet example

This end-to-end example shows how to:
* build `Resnet 18 v1` with `tvm` and `nnvm`. 
* use the provided Rust frontend API to test for an input image. 

To run the example, first `tvm`, `nnvm` and `mxnet` must be installed for the python build. To install mxnet, run `pip install mxnet`
and to install `tvm` and `nnvm` with `llvm` follow the [TVM installation guide](https://docs.tvm.ai/install/index.html).

* **Build**: `cargo build`

* **Run**: `cargo run`
