#![feature(extern_crate_item_prelude, panic_info_message)]
#![allow(unused_imports, unused_must_use)]

use std::panic;

extern crate tvm_rust as tvm;

use tvm::*;

fn main() {
    register_global_func! {
        fn error(_args: &[TVMArgValue]) -> Result<TVMRetValue> {
            Err(Error::TypeMismatch {
                expected: "i64".to_string(),
                found: "f64".to_string(),
            })
        }
    }

    let mut registered = function::Builder::default();
    registered.get_function("error", true, false);
    assert!(registered.func.is_some());
    registered.args(&[10, 20]);

    panic::set_hook(Box::new(|panic_info| {
        if let Some(msg) = panic_info.message() {
            println!("{:?}", msg);
        }
        if let Some(location) = panic_info.location() {
            println!(
                "panic occurred in file '{}' at line {}",
                location.file(),
                location.line()
            );
        } else {
            println!("panic occurred but can't get location information");
        }
    }));
    let _result = panic::catch_unwind(move || registered.invoke());
    // neither `AssertUnwindSafe` nor `assert!(result.is_err())` work!
}
