use custom_error::custom_error;

custom_error! {pub Error
    EmptyArray = "Cannot convert from empty array",
    NullHandle { name: String } = "Null handle for {name}",
    NoFunction = "Function was not set in function::Builder.",
    TypeMismatch { expected: String, found: String } = "Expected type {expected}, but found {found}",
}

pub type Result<T> = ::std::result::Result<T, Error>;
