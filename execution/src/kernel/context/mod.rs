mod cpu;

pub use cpu::CpuKernelContext;

use crate::ExecutionTag;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelContext {
    Cpu(CpuKernelContext),
}

impl KernelContext {
    pub fn cpu() -> Self {
        Self::Cpu(CpuKernelContext::new())
    }

    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }
}
