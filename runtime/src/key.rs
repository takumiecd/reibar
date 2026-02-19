use execution::ExecutionTag;

use crate::OpTag;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKey {
    execution: ExecutionTag,
    op: OpTag,
}

impl KernelKey {
    pub fn new(execution: ExecutionTag, op: OpTag) -> Self {
        Self { execution, op }
    }

    pub fn cpu(op: OpTag) -> Self {
        Self::new(ExecutionTag::Cpu, op)
    }

    pub fn execution(&self) -> ExecutionTag {
        self.execution
    }

    pub fn op(&self) -> OpTag {
        self.op
    }
}
