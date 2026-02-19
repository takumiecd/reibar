use crate::CpuExecution;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTag {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Execution {
    Cpu(CpuExecution),
}

impl Execution {
    pub fn cpu() -> Self {
        Self::Cpu(CpuExecution)
    }

    pub fn tag(&self) -> ExecutionTag {
        match self {
            Self::Cpu(_) => ExecutionTag::Cpu,
        }
    }
}

impl ExecutionTag {
    pub fn execution(self) -> Execution {
        match self {
            Self::Cpu => Execution::cpu(),
        }
    }
}
