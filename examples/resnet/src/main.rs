extern crate csv;
extern crate image;
extern crate ndarray;
extern crate tvm_rust as tvm;

use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::path::Path;
use std::result::Result;

use image::{FilterType, GenericImageView};
use ndarray::{Array, ArrayD, Axis};

use tvm::*;

fn main() -> Result<(), Box<Error>> {
    let ctx = TVMContext::cpu(0);
    let img = image::open("cat.png")?;
    println!("originam image dimensions: {:?}", img.dimensions());
    let img = img.resize(224, 224, FilterType::Nearest).to_rgb();
    println!("resized image dimensions: {:?}", img.dimensions());
    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        let tmp = [
            (tmp[0] as f32 - 123.0) / 58.395, // R
            (tmp[1] as f32 - 117.0) / 57.12,  // G
            (tmp[2] as f32 - 104.0) / 57.375, // B
        ];
        for e in &tmp {
            pixels.push(*e);
        }
    }

    let arr = Array::from_shape_vec((224, 224, 3), pixels)?;
    let arr: ArrayD<f32> = arr.permuted_axes([2, 0, 1]).into_dyn();
    let arr = arr.insert_axis(Axis(0));
    let input = NDArray::from_rust_ndarray(&arr, TVMContext::cpu(0), TVMType::from("float"))?;

    let graph = fs::read_to_string("deploy_graph.json")?;

    let lib = Module::load(&Path::new("deploy_lib.so"))?;

    let runtime_create_fn =
        Function::get_function("tvm.graph_runtime.create".to_string(), true, false).unwrap();

    let runtime_create_fn_ret = function::Builder::from(runtime_create_fn)
        .arg(&graph)
        .arg(&lib)
        .arg(&ctx.device_type)
        .arg(&ctx.device_id)
        .invoke()?;

    let graph_runtime_module = runtime_create_fn_ret.to_module();

    let load_param_fn = graph_runtime_module
        .get_function("load_params", false)
        .unwrap();

    let params: Vec<u8> = fs::read("deploy_param.params")?;
    let barr = TVMByteArray::from(&params);

    function::Builder::from(load_param_fn).arg(&barr).invoke()?;

    let set_input_fn = graph_runtime_module
        .get_function("set_input", false)
        .unwrap();

    function::Builder::from(set_input_fn)
        .arg("data")
        .arg(&input)
        .invoke()?;

    let run_fn = graph_runtime_module.get_function("run", false).unwrap();

    function::Builder::from(run_fn).invoke()?;

    let mut output_shape = vec![1, 1000];
    let output = empty(
        &mut output_shape,
        TVMContext::cpu(0),
        TVMType::from("float"),
    );

    let get_output_fn = graph_runtime_module
        .get_function("get_output", false)
        .unwrap();

    function::Builder::from(get_output_fn)
        .arg(&0)
        .arg(&output)
        .invoke()?;

    let output = output.to_vec::<f32>()?;

    let mut argmax = -1;
    let mut max_prob = 0.;
    for i in 0..output.len() {
        if output[i] > max_prob {
            max_prob = output[i];
            argmax = i as i32;
        }
    }

    let mut synset: HashMap<i32, String> = HashMap::new();

    let file = File::open("./synset.csv")?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    for result in rdr.records() {
        let record = result?;
        let id: i32 = record[0].parse()?;
        let cls = record[1].to_string();
        synset.insert(id, cls);
    }

    println!(
        "input belongs to class `{}` with probability {}",
        synset.get(&argmax).unwrap(),
        max_prob
    );

    Ok(())
}
