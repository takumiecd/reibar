mod cpu;

pub use cpu::CpuKernelArgs;

use crate::ExecutionTag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelArgs {
    Cpu(CpuKernelArgs),
}

impl KernelArgs {
    pub fn cpu(args: CpuKernelArgs) -> Self {
        Self::Cpu(args)
    }

    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }
}
