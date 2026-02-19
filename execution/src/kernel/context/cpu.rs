#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernelContext {
    worker_threads: usize,
}

impl CpuKernelContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_worker_threads(mut self, worker_threads: usize) -> Self {
        self.worker_threads = worker_threads.max(1);
        self
    }

    pub fn worker_threads(&self) -> usize {
        self.worker_threads
    }
}

impl Default for CpuKernelContext {
    fn default() -> Self {
        Self { worker_threads: 1 }
    }
}
