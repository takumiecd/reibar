#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuKernelArgs {
    storage_count: usize,
    param_count: usize,
}

impl CpuKernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_storage_count(mut self, value: usize) -> Self {
        self.storage_count = value;
        self
    }

    pub fn with_param_count(mut self, value: usize) -> Self {
        self.param_count = value;
        self
    }

    pub fn storage_count(&self) -> usize {
        self.storage_count
    }

    pub fn param_count(&self) -> usize {
        self.param_count
    }
}
