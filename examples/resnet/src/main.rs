extern crate csv;
extern crate image;
extern crate ndarray;
extern crate tvm_frontend as tvm;

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
    println!("original image dimensions: {:?}", img.dimensions());
    // for bigger size images, one needs to first resize to 256x256
    // with `img.resize_exact` method and then crop to 224x224
    let img = img.resize(224, 224, FilterType::Nearest).to_rgb();
    println!("resized image dimensions: {:?}", img.dimensions());
    let mut pixels: Vec<f32> = vec![];
    for pixel in img.pixels() {
        let tmp = pixel.data;
        // normalize the RGB channels using mean, std of imagenet1k
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
    // create input tensor from rust's ndarray
    let input = NDArray::from_rust_ndarray(&arr, TVMContext::cpu(0), TVMType::from("float"))?;
    println!("input size is {:?}", input.shape().unwrap());
    let graph = fs::read_to_string("deploy_graph.json")?;
    // load module
    let lib = Module::load(&Path::new("deploy_lib.so"))?;
    // get the global TVM graph runtime function
    let runtime_create_fn =
        Function::get_function("tvm.graph_runtime.create", true, false).unwrap();
    // create runtime function from Rust
    let runtime_create_fn_ret = function::Builder::from(runtime_create_fn)
        .arg(&graph)
        .arg(&lib)
        .arg(&ctx.device_type)
        .arg(&ctx.device_id)
        .invoke()?;
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
    function::Builder::from(load_param_fn).arg(&barr).invoke()?;
    // get the set_input function
    let set_input_fn = graph_runtime_module
        .get_function("set_input", false)
        .unwrap();
    // set the input via set_input function
    function::Builder::from(set_input_fn)
        .arg("data")
        .arg(&input)
        .invoke()?;
    // get `run` function from runtime module
    let run_fn = graph_runtime_module.get_function("run", false).unwrap();
    // execute the run function
    function::Builder::from(run_fn).invoke()?;
    // prepare to get the output
    let mut output_shape = vec![1, 1000];
    let output = empty(
        &mut output_shape,
        TVMContext::cpu(0),
        TVMType::from("float"),
    );
    // get the `get_output` function from runtime module
    let get_output_fn = graph_runtime_module
        .get_function("get_output", false)
        .unwrap();
    // execute the get output function
    function::Builder::from(get_output_fn)
        .arg(&0)
        .arg(&output)
        .invoke()?;
    // flatten the output as Vec<f32>
    let output = output.to_vec::<f32>()?;
    // find the maximum entry in the output and its index
    let mut argmax = -1;
    let mut max_prob = 0.;
    for i in 0..output.len() {
        if output[i] > max_prob {
            max_prob = output[i];
            argmax = i as i32;
        }
    }
    // create a hash map of (class id, class name)
    let mut synset: HashMap<i32, String> = HashMap::new();

    let file = File::open("synset.csv")?;
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
        "input image belongs to the class `{}` with probability {}",
        synset.get(&argmax).unwrap(),
        max_prob
    );

    Ok(())
}
