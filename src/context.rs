use super::*;

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

// TODO: include extTypes
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

// TODO: reexport devs
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

    pub fn cpu(device_id: i32) -> Self {
        TVMContext {
            device_id: device_id,
            ..Default::default()
        }
    }
    // TODO: refactor to macro impl
    pub fn gpu(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLGPU), device_id)
    }

    pub fn cpu_pinned(device_id: i32) -> Self {
        Self::new(
            TVMDeviceType::new(tvm::DLDeviceType::kDLCPUPinned),
            device_id,
        )
    }

    pub fn opencl(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLOpenCL), device_id)
    }

    pub fn metal(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLMetal), device_id)
    }

    pub fn vpi(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLVPI), device_id)
    }

    pub fn rocm(device_id: i32) -> Self {
        Self::new(TVMDeviceType::new(tvm::DLDeviceType::kDLROCM), device_id)
    }
}

impl TVMContext {
    // pub fn exist(&self) -> bool {
    //     let func = get_api("_GetDeviceAttr".to_owned(), true);
    //     let dt = self.device_type.inner as i32;
    //     let ret = func
    //         .push_arg(&dt)
    //         .push_arg(&self.device_id)
    //         .push_arg(&0)
    //         .invoke();
    //     ret.value != TVMValue::default()
    // }

    // pub fn max_thread_pre_block(&self) -> usize {
    //     let func = get_api("_GetDeviceAttr".to_owned(), true);
    //     let dt = self.device_type.inner as i32;
    //     let ret = func
    //         .push_arg(&dt)
    //         .push_arg(&self.device_id)
    //         .push_arg(&1)
    //         .invoke();
    //     unsafe { ret.value.inner.v_int64 as usize }
    // }

    // pub fn warp_size(&self) -> usize {
    //     let func = get_api("_GetDeviceAttr".to_owned(), true);
    //     let dt = self.device_type.inner as i32;
    //     let ret = func
    //         .push_arg(&dt)
    //         .push_arg(&self.device_id)
    //         .push_arg(&2)
    //         .invoke();
    //     unsafe { ret.value.inner.v_int64 as usize }
    // }

    pub fn sync(&self) -> TVMResult<()> {
        //let handle = ptr::null_mut();
        check_call!(tvm::TVMSynchronize(
            self.device_type.inner as i32,
            self.device_id,
            ptr::null_mut() as *mut c_void
        ));
        Ok(())
    }
}

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
        // TODO: display DeviceType
        write!(f, "{:?} {}", self.device_type, self.device_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context() {
        let ctx = TVMContext::cpu(0);
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

    // #[test]
    // fn dev_attribute() {
    //     let ctx = TVMContext::cpu(0);
    //     assert!(ctx.exist());
    //     println!("max thread per block: {}", ctx.max_thread_pre_block());
    //     println!("warp size: {}", ctx.warp_size());
    // }
}
