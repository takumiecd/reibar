mod args;
mod context;
mod cpu;

pub use args::{CpuKernelArgs, KernelArgs};
pub use context::{CpuKernelContext, KernelContext};
pub use cpu::{CpuKernelFn, CpuKernelLauncher, CpuKernelMetadata};

use crate::ExecutionTag;

#[derive(Debug, Clone)]
pub enum KernelMetadata {
    Cpu(CpuKernelMetadata),
}

impl KernelMetadata {
    pub fn cpu(name: impl Into<String>, kernel_fn: CpuKernelFn) -> Self {
        Self::Cpu(CpuKernelMetadata::new(name, kernel_fn))
    }

    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }

    pub fn into_launcher(self) -> KernelLauncher {
        match self {
            Self::Cpu(metadata) => KernelLauncher::Cpu(metadata.into_launcher()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum KernelLauncher {
    Cpu(CpuKernelLauncher),
}

impl KernelLauncher {
    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }

    pub fn launch(&self, args: &KernelArgs) -> Result<(), KernelLaunchError> {
        match self {
            Self::Cpu(launcher) => match args {
                KernelArgs::Cpu(cpu_args) => launcher.launch(cpu_args),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelLaunchError {
    ExecutionMismatch {
        launcher: ExecutionTag,
        args: ExecutionTag,
    },
}
