mod cpu;

pub use cpu::CpuStorage;

use crate::ExecutionTag;

#[derive(Debug, Clone, PartialEq)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Cpu(storage) => storage.len(),
        }
    }
}
