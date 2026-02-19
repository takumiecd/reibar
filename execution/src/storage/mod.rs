mod cpu;

pub use cpu::CpuStorage;

use crate::ExecutionTag;

#[derive(Debug, Clone, PartialEq)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn from_vec(tag: ExecutionTag, data: Vec<f32>) -> Self {
        match tag {
            ExecutionTag::Cpu => Self::cpu_from_vec(data),
        }
    }

    pub fn filled(tag: ExecutionTag, len: usize, value: f32) -> Self {
        match tag {
            ExecutionTag::Cpu => Self::cpu_filled(len, value),
        }
    }

    pub fn zeros(tag: ExecutionTag, len: usize) -> Self {
        Self::filled(tag, len, 0.0)
    }

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

    pub fn as_slice(&self) -> &[f32] {
        match self {
            Self::Cpu(storage) => storage.as_slice(),
        }
    }

    pub(crate) fn cpu_from_vec(data: Vec<f32>) -> Self {
        Self::Cpu(CpuStorage::from_vec(data))
    }

    pub(crate) fn cpu_filled(len: usize, value: f32) -> Self {
        Self::Cpu(CpuStorage::filled(len, value))
    }
}
