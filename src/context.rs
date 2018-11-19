use std::fmt::{self, Display, Formatter};
use std::os::raw::c_void;
use std::ptr;

use ts;

use function;
use internal_api;
use TVMResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TVMDeviceType(pub usize);

impl Default for TVMDeviceType {
    fn default() -> Self {
        TVMDeviceType(1)
    }
}

impl From<ts::DLDeviceType> for TVMDeviceType {
    fn from(device_type: ts::DLDeviceType) -> Self {
        match device_type {
            ts::DLDeviceType::kDLCPU => TVMDeviceType(1),
            ts::DLDeviceType::kDLGPU => TVMDeviceType(2),
            ts::DLDeviceType::kDLCPUPinned => TVMDeviceType(3),
            ts::DLDeviceType::kDLOpenCL => TVMDeviceType(4),
            ts::DLDeviceType::kDLMetal => TVMDeviceType(8),
            ts::DLDeviceType::kDLVPI => TVMDeviceType(9),
            ts::DLDeviceType::kDLROCM => TVMDeviceType(10),
        }
    }
}

impl Display for TVMDeviceType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            TVMDeviceType(1) => write!(f, "cpu"),
            TVMDeviceType(2) => write!(f, "gpu"),
            TVMDeviceType(3) => write!(f, "cpu_pinned"),
            TVMDeviceType(4) => write!(f, "opencl"),
            TVMDeviceType(8) => write!(f, "meta"),
            TVMDeviceType(9) => write!(f, "vpi"),
            TVMDeviceType(10) => write!(f, "rocm"),
            TVMDeviceType(_) => write!(f, "rpc"),
        }
    }
}

impl<'a> From<&'a str> for TVMDeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => TVMDeviceType(1),
            "llvm" => TVMDeviceType(1),
            "stackvm" => TVMDeviceType(1),
            "gpu" => TVMDeviceType(2),
            "cuda" => TVMDeviceType(2),
            "nvptx" => TVMDeviceType(2),
            "cl" => TVMDeviceType(4),
            "opencl" => TVMDeviceType(4),
            "metal" => TVMDeviceType(8),
            "vpi" => TVMDeviceType(9),
            "rocm" => TVMDeviceType(10),
            _ => panic!("{:?} not supported!", type_str),
        }
    }
}

/// TVM context. Default is cpu.
#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
pub struct TVMContext {
    /// Supported devices
    pub device_type: TVMDeviceType,
    /// Device id
    pub device_id: i32,
}

impl TVMContext {
    pub fn new(device_type: TVMDeviceType, device_id: i32) -> Self {
        TVMContext {
            device_type: device_type,
            device_id: device_id,
        }
    }

    pub fn current_context(&self) -> &Self {
        self
    }
}

macro_rules! impl_ctx {
    ($ctx:ident, $dldevt:expr) => {
        impl TVMContext {
            pub fn $ctx(device_id: i32) -> Self {
                Self::new(TVMDeviceType($dldevt), device_id)
            }
        }
    };
}

impl_ctx!(cpu, 1);
impl_ctx!(gpu, 2);
impl_ctx!(cpu_pinned, 3);
impl_ctx!(cuda, 2);
impl_ctx!(nvptx, 2);
impl_ctx!(cl, 4);
impl_ctx!(opencl, 4);
impl_ctx!(metal, 8);
impl_ctx!(vpi, 9);
impl_ctx!(rocm, 10);

impl<'a> From<&'a str> for TVMContext {
    fn from(target: &str) -> Self {
        TVMContext::new(TVMDeviceType::from(target), 0)
    }
}

impl TVMContext {
    pub fn exist(&self) -> bool {
        let func = internal_api::get_api("_GetDeviceAttr".to_owned());
        let dt = self.device_type.0 as i32;
        let ret = function::Builder::from(func)
            .arg(&dt)
            .arg(&self.device_id)
            .arg(&0)
            .invoke()
            .unwrap();
        ret.to_int() != 0
    }

    pub fn sync(&self) -> TVMResult<()> {
        check_call!(ts::TVMSynchronize(
            self.device_type.0 as i32,
            self.device_id,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }
}

macro_rules! impl_dev_attrs {
    ($attr_name:ident, $attr_kind:expr) => {
        impl TVMContext {
            pub fn $attr_name(&self) -> usize {
                let func = ::internal_api::get_api("_GetDeviceAttr".to_owned());
                let dt = self.device_type.0 as i32;
                let ret = function::Builder::from(func)
                    .args(&[dt, self.device_id, $attr_kind])
                    .invoke()
                    .unwrap();
                ret.to_int() as usize
            }
        }
    };
}

impl_dev_attrs!(max_threads_per_block, 1);
impl_dev_attrs!(warp_size, 2);
impl_dev_attrs!(max_shared_memory_per_block, 3);
impl_dev_attrs!(compute_version, 4);
impl_dev_attrs!(device_name, 5);
impl_dev_attrs!(max_clock_rate, 6);
impl_dev_attrs!(multi_processor_count, 7);
impl_dev_attrs!(max_thread_dimensions, 8);

impl From<ts::DLContext> for TVMContext {
    fn from(ctx: ts::DLContext) -> Self {
        TVMContext {
            device_type: TVMDeviceType::from(ctx.device_type),
            device_id: ctx.device_id,
        }
    }
}

impl Display for TVMContext {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}({})", self.device_type, self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = TVMContext::cpu(0);
        println!("ctx: {}", ctx);
        let default_ctx = TVMContext::new(TVMDeviceType(1), 0);
        assert_eq!(ctx.current_context().clone(), default_ctx);
        assert_ne!(ctx, TVMContext::gpu(0));

        let str_ctx = TVMContext::new(TVMDeviceType::from("gpu"), 0);
        assert_eq!(str_ctx.current_context().clone(), str_ctx);
        assert_ne!(str_ctx, TVMContext::new(TVMDeviceType::from("cpu"), 0));
    }

    #[test]
    fn sync() {
        let ctx = TVMContext::cpu(0);
        assert!(ctx.sync().is_ok())
    }

    #[test]
    fn dev_attributes() {
        let ctx = TVMContext::cpu(0);
        assert!(ctx.exist());
        println!("max thread per block: {}", ctx.max_threads_per_block());
        println!("warp size: {}", ctx.warp_size());
        println!(
            "max shared memory per block: {}",
            ctx.max_shared_memory_per_block()
        );
        println!("compute version: {}", ctx.compute_version());
        println!("device name: {}", ctx.device_name());
        println!("max clock rate: {}", ctx.max_clock_rate());
        println!("multi processor count: {}", ctx.multi_processor_count());
        println!("max thread dimensions: {}", ctx.max_thread_dimensions());
    }
}
