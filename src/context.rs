use std::fmt::{self, Formatter, Display};
use std::os::raw::c_void;
use std::ptr;

use tvm;

use internal_api;
use function;
use TVMValue;
use TVMResult;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct TVMDeviceType {
    pub(crate) inner: tvm::DLDeviceType,
}

impl TVMDeviceType {
    pub(crate) fn new(device_type: tvm::DLDeviceType) -> Self {
        TVMDeviceType { inner: device_type }
    }
}

impl Default for TVMDeviceType {
    fn default() -> Self {
        TVMDeviceType::new(tvm::DLDeviceType::kDLCPU)
    }
}

impl From<TVMDeviceType> for tvm::DLDeviceType {
    fn from(device_type: TVMDeviceType) -> Self {
        device_type.inner
    }
}

impl From<tvm::DLDeviceType> for TVMDeviceType {
    fn from(device_type: tvm::DLDeviceType) -> Self {
        TVMDeviceType::new(device_type)
    }
}

impl Display for TVMDeviceType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self.inner {
            tvm::DLDeviceType::kDLCPU => write!(f, "cpu"),
            tvm::DLDeviceType::kDLCPUPinned => write!(f, "cpu_pinned"),
            tvm::DLDeviceType::kDLGPU => write!(f, "gpu"),
            tvm::DLDeviceType::kDLOpenCL => write!(f, "opencl"),
            tvm::DLDeviceType::kDLMetal => write!(f, "meta"),
            tvm::DLDeviceType::kDLVPI => write!(f, "vpi"),
            tvm::DLDeviceType::kDLROCM => write!(f, "rocm"),
        }
    }
}

impl<'a> From<&'a str> for TVMDeviceType {
    fn from(type_str: &'a str) -> Self {
        match type_str {
            "cpu" => Self::new(tvm::DLDeviceType::kDLCPU),
            "llvm" => Self::new(tvm::DLDeviceType::kDLCPU),
            "stackvm" => Self::new(tvm::DLDeviceType::kDLCPU),
            "gpu" => Self::new(tvm::DLDeviceType::kDLGPU),
            "cuda" => Self::new(tvm::DLDeviceType::kDLGPU),
            "nvptx" => Self::new(tvm::DLDeviceType::kDLGPU),
            "cl" => Self::new(tvm::DLDeviceType::kDLOpenCL),
            "opencl" => Self::new(tvm::DLDeviceType::kDLOpenCL),
            "metal" => Self::new(tvm::DLDeviceType::kDLMetal),
            "vpi" => Self::new(tvm::DLDeviceType::kDLVPI),
            "rocm" => Self::new(tvm::DLDeviceType::kDLROCM),
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

macro_rules! impl_context {
    ($ctx:ident, $dldevt:expr) => {
        impl TVMContext {
            pub fn $ctx(device_id: i32) -> Self {
                Self::new(TVMDeviceType::new($dldevt), device_id)
            }
        }
    }
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

impl_context!(cpu, tvm::DLDeviceType::kDLCPU);
impl_context!(cpu_pinned, tvm::DLDeviceType::kDLCPUPinned);
impl_context!(gpu, tvm::DLDeviceType::kDLGPU);
impl_context!(cuda, tvm::DLDeviceType::kDLGPU);
impl_context!(nvptx, tvm::DLDeviceType::kDLGPU);
impl_context!(cl, tvm::DLDeviceType::kDLOpenCL);
impl_context!(opencl, tvm::DLDeviceType::kDLOpenCL);
impl_context!(metal, tvm::DLDeviceType::kDLMetal);
impl_context!(vpi, tvm::DLDeviceType::kDLVPI);
impl_context!(rocm, tvm::DLDeviceType::kDLROCM);

macro_rules! impl_dev_attrs {
    ($attr_name:ident, $attr_kind:expr) => {
        impl TVMContext {
            pub fn $attr_name(&self) -> usize {
                let func = ::internal_api::get_api("_GetDeviceAttr".to_owned());
                let dt = self.device_type.inner as i32;
                let ret = function::Builder::from(func)
                    .push_arg(&dt)
                    .push_arg(&self.device_id)
                    .push_arg(&$attr_kind)
                    .invoke()
                    .unwrap();
                unsafe { ret.value.inner.v_int64 as usize }
            }
        }
    }
}

impl<'a> From<&'a str> for TVMContext {
    fn from(target: &str) -> Self {
        TVMContext::new(TVMDeviceType::from(target), 0)
    }
}

impl TVMContext {
    pub fn exist(&self) -> bool {
        let func = internal_api::get_api("_GetDeviceAttr".to_owned());
        let dt = self.device_type.inner as i32;
        let ret = function::Builder::from(func)
            .push_arg(&dt)
            .push_arg(&self.device_id)
            .push_arg(&0)
            .invoke()
            .unwrap();
        ret.value != TVMValue::default()
    }

    pub fn sync(&self) -> TVMResult<()> {
        check_call!(tvm::TVMSynchronize(
            self.device_type.inner as i32,
            self.device_id,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }
}

impl_dev_attrs!(max_threads_per_block, 1);
impl_dev_attrs!(warp_size, 2);
impl_dev_attrs!(max_shared_memory_per_block, 3);
impl_dev_attrs!(compute_version, 4);
impl_dev_attrs!(device_name, 5);
impl_dev_attrs!(max_clock_rate, 6);
impl_dev_attrs!(multi_processor_count, 7);
impl_dev_attrs!(max_thread_dimensions, 8);

impl From<TVMContext> for tvm::DLContext {
    fn from(ctx: TVMContext) -> Self {
        tvm::DLContext {
            device_type: tvm::DLDeviceType::from(ctx.device_type),
            device_id: ctx.device_id,
        }
    }
}

impl From<tvm::DLContext> for TVMContext {
    fn from(ctx: tvm::DLContext) -> Self {
        TVMContext {
            device_type: ctx.device_type.into(),
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
        let default_ctx = TVMContext::new(TVMDeviceType::new(tvm::DLDeviceType::kDLCPU), 0);
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
        println!("max shared memory per block: {}", ctx.max_shared_memory_per_block());
        println!("compute version: {}", ctx.compute_version());
        println!("device name: {}", ctx.device_name());
        println!("max clock rate: {}", ctx.max_clock_rate());
        println!("multi processor count: {}", ctx.multi_processor_count());
        println!("max thread dimensions: {}", ctx.max_thread_dimensions());
    }
}
