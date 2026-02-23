#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuStorageContext {
    numa_node: usize,
}

impl CpuStorageContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_numa_node(mut self, numa_node: usize) -> Self {
        self.numa_node = numa_node;
        self
    }

    pub fn numa_node(&self) -> usize {
        self.numa_node
    }
}

impl execution_contracts::ExecutionStorageContext for CpuStorageContext {}
