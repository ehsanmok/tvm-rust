//! This module implements the TVM custom [`Error`] and [`Result`] types.

// use std::option;
use std::ffi;

error_chain!{
    errors {
        EmptyArray {
            description("cannot convert from empty array")
        }
        NullHandle(name: String) {
            description("null handle")
            display("requested `{}` handle is null", name)
        }
        NoFunction {
            description("function was not set in `function::Builder`")
        }
        TypeMismatch(expected: String, found: String) {
            description("type mismatch!")
            display("expected type `{}`, but found `{}`", expected, found)
        }

    }
    foreign_links {
        // NoneError(option::NoneError);
        CStringNulError(ffi::NulError);
        CStringIntoString(ffi::IntoStringError);
    }
}
