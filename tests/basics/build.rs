use std::process::Command;
use std::env;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let script_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/tvm_add_cpu.py");
    let output = Command::new("python")
      .args(&[&script_path, &out_dir.as_str()])
      .output()
      .expect("Failed to execute command");
    if output.stderr.len() > 0 {
        panic!(String::from_utf8(output.stderr).unwrap());
    }
    println!("cargo:rustc-link-search=native={}", script_path);
}
