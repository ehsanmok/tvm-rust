extern crate bindgen;

use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process;
use std::result::Result;

const TVM_RUNTIME: &'static str = "tvm_runtime";

fn main() {
    match run() {
        Ok(_) => (),
        Err(err) => {
            eprintln!("error occured during the build: {:?}", err);
            process::exit(1)
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rustc-link-lib=dylib={}", TVM_RUNTIME);
    let lib = format!("lib{}", TVM_RUNTIME);
    println!("cargo:rustc-link-search=native={}", lib);
    let tvm_home = env::var("TVM_HOME").expect("TVM_HOME not found!");
    let bindings = bindgen::Builder::default()
        .header(format!("{}/include/tvm/runtime/c_runtime_api.h", tvm_home))
        .clang_arg(format!("-I{}/3rdparty/dlpack/include/", tvm_home))
        .rustified_enum("DLDataTypeCode")
        .rustified_enum("DLDeviceType")
        .rustified_enum("TVMTypeCode")
        .rustified_enum("TVMDeviceExtType")
        .blacklist_type("max_align_t")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(PathBuf::from("src/bindgen.rs"))
        .expect("Can not write the bindings!");

    Ok(())
}
