use schema::{KernelArg, KernelArgs, KernelArgsError};

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

#[derive(Debug, Clone, PartialEq)]
pub struct CpuKernelArgs {
    context: CpuKernelContext,
    args: KernelArgs<()>,
}

impl CpuKernelArgs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: CpuKernelContext) -> Self {
        self.context = context;
        self
    }

    pub fn context(&self) -> &CpuKernelContext {
        &self.context
    }

    pub fn args(&self) -> &KernelArgs<()> {
        &self.args
    }

    pub fn args_mut(&mut self) -> &mut KernelArgs<()> {
        &mut self.args
    }

    pub fn insert(&mut self, arg: KernelArg<()>) -> Result<(), KernelArgsError> {
        self.args.insert(arg)
    }

    pub fn len(&self) -> usize {
        self.args.len()
    }

}

impl Default for CpuKernelArgs {
    fn default() -> Self {
        Self {
            context: CpuKernelContext::default(),
            args: KernelArgs::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernelLaunchError {
    message: String,
}

impl CpuKernelLaunchError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

pub type CpuKernelFn = fn(&CpuKernelArgs) -> Result<(), CpuKernelLaunchError>;

#[derive(Debug, Clone)]
pub struct CpuKernelMetadata {
    kernel_name: String,
    kernel_fn: CpuKernelFn,
}

impl CpuKernelMetadata {
    pub fn new(name: impl Into<String>, kernel_fn: CpuKernelFn) -> Self {
        Self {
            kernel_name: name.into(),
            kernel_fn,
        }
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn kernel_fn(&self) -> CpuKernelFn {
        self.kernel_fn
    }

    pub fn into_launcher(self) -> CpuKernelLauncher {
        CpuKernelLauncher {
            kernel_name: self.kernel_name,
            kernel_fn: self.kernel_fn,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuKernelLauncher {
    kernel_name: String,
    kernel_fn: CpuKernelFn,
}

impl CpuKernelLauncher {
    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn kernel_fn(&self) -> CpuKernelFn {
        self.kernel_fn
    }

    pub fn to_metadata(&self) -> CpuKernelMetadata {
        CpuKernelMetadata::new(self.kernel_name.clone(), self.kernel_fn)
    }

    pub fn launch(&self, args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
        (self.kernel_fn)(args)
    }
}
