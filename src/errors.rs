//! This module implements TVM custom [`Error`], [`ErrorKind`] and [`Result`] types.

use std::{ffi, option};

use rust_ndarray;

error_chain!{
    errors {
        EmptyArray {
            description("cannot convert from empty array")
        }
        NullHandle(name: String) {
            description("null handle")
            display("requested `{}` handle is null", name)
        }
        FunctionNotFound {
            description("function not found")
            display("function was not set in `function::Builder`")
        }
        TypeMismatch(expected: String, found: String) {
            description("type mismatch!")
            display("expected type `{}`, but found `{}`", expected, found)
        }
        MissingShapeError {
            description("ndarray `shape()` returns `None`")
            display("called `Option::unwrap()` on a `None` value")
        }

    }
    foreign_links {
        ShapeError(rust_ndarray::ShapeError);
        NulError(ffi::NulError);
        IntoStringError(ffi::IntoStringError);
    }
}

impl From<option::NoneError> for Error {
    fn from(err: option::NoneError) -> Self {
        ErrorKind::MissingShapeError.into()
    }
}
