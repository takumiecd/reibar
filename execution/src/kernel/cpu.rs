use super::{CpuKernelArgs, KernelLaunchError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernelMetadata {
    kernel_name: String,
}

impl CpuKernelMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            kernel_name: name.into(),
        }
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn into_launcher(self) -> CpuKernelLauncher {
        CpuKernelLauncher {
            kernel_name: self.kernel_name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernelLauncher {
    kernel_name: String,
}

impl CpuKernelLauncher {
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn launch(&self, _args: &CpuKernelArgs) -> Result<(), KernelLaunchError> {
        // Placeholder: concrete kernel invocation will be wired by dispatcher/registry.
        Ok(())
    }
}
