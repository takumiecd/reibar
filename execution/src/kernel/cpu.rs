use super::{CpuKernelArgs, KernelLaunchError};

pub type CpuKernelFn = fn(&CpuKernelArgs) -> Result<(), KernelLaunchError>;

#[derive(Debug, Clone)]
pub struct CpuKernelMetadata {
    kernel_name: String,
    kernel_fn: CpuKernelFn,
}

impl CpuKernelMetadata {
    pub fn new(name: impl Into<String>, kernel_fn: CpuKernelFn) -> Self {
        Self {
            kernel_name: name.into(),
            kernel_fn,
        }
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn kernel_fn(&self) -> CpuKernelFn {
        self.kernel_fn
    }

    pub fn into_launcher(self) -> CpuKernelLauncher {
        CpuKernelLauncher {
            kernel_name: self.kernel_name,
            kernel_fn: self.kernel_fn,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuKernelLauncher {
    kernel_name: String,
    kernel_fn: CpuKernelFn,
}

impl CpuKernelLauncher {
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn kernel_fn(&self) -> CpuKernelFn {
        self.kernel_fn
    }

    pub fn to_metadata(&self) -> CpuKernelMetadata {
        CpuKernelMetadata::new(self.kernel_name.clone(), self.kernel_fn)
    }

    pub fn launch(&self, args: &CpuKernelArgs) -> Result<(), KernelLaunchError> {
        (self.kernel_fn)(args)
    }
}
