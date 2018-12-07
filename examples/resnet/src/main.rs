#![allow(unused_imports, unused_variables, unused_mut)]

extern crate image;
extern crate ndarray;
extern crate tvm_rust as tvm;

use std::error::Error;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::Path;
use std::result::Result;

use image::{FilterType, GenericImageView};
use ndarray::{Array, ArrayD};

use tvm::*;

fn main() -> Result<(), Box<Error>> {
    let ctx = TVMContext::cpu(0);
    let img = image::open("cat.jpeg")?;
    println!("image dimensions: {:?}", img.dimensions());
    let img = img.resize(224, 224, FilterType::Nearest).to_rgb();
    println!("image dimensions: {:?}", img.dimensions());
    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        let tmp = [
            (tmp[0] as f32 - 120.45) / 127.5,
            (tmp[0] as f32 - 115.74) / 127.5,
            (tmp[0] as f32 - 104.65) / 127.5,
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    let arr = Array::from_shape_vec((1, 3, 224, 224), pixels)?;
    let arr: ArrayD<f32> = arr.into_dyn();
    let input = NDArray::from_rust_ndarray(&arr, TVMContext::cpu(0), TVMType::from("float"))?;

    let graph = fs::read_to_string("deploy_graph.json")?;
    // println!("{:?}", graph);

    let lib = Module::load(&Path::new("deploy_lib.so"))?;

    let runtime_create_fn =
        Function::get_function("tvm.graph_runtime.create".to_string(), true, false).unwrap();

    let runtime_create_fn_ret = function::Builder::from(runtime_create_fn)
        .arg(&graph)
        .arg(&lib)
        .arg(&ctx.device_type)
        .arg(&ctx.device_id)
        .invoke()?;

    let mut graph_runtime_module = runtime_create_fn_ret.to_module();

    let load_param_fn = graph_runtime_module
        .get_function("load_params", false)
        .unwrap();

    let mut params: Vec<u8> = fs::read("deploy_param.params")?;

    let params = TVMByteArray::from(&params);

    let mut ret = function::Builder::from(load_param_fn);
    ret.arg(&params).invoke()?;
    println!("{:?}", ret);

    Ok(())
}
