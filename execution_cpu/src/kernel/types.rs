use schema::{KernelArg, KernelArgs, KernelArgsError};

use crate::storage::CpuStorage;

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

#[derive(Debug, Clone)]
pub struct CpuKernelArgs {
    context: CpuKernelContext,
    args: KernelArgs<CpuStorage>,
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

    pub fn args(&self) -> &KernelArgs<CpuStorage> {
        &self.args
    }

    pub fn args_mut(&mut self) -> &mut KernelArgs<CpuStorage> {
        &mut self.args
    }

    pub fn insert(&mut self, arg: KernelArg<CpuStorage>) -> Result<(), KernelArgsError> {
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CpuKernelEntrypoint {
    device_function: Option<String>,
}

impl CpuKernelEntrypoint {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_device_function(mut self, device_function: impl Into<String>) -> Self {
        self.device_function = Some(device_function.into());
        self
    }

    pub fn device_function(&self) -> Option<&str> {
        self.device_function.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuKernelLaunchConfig {
    kernel_name: String,
    entrypoint: CpuKernelEntrypoint,
}

impl CpuKernelLaunchConfig {
    pub fn new(kernel_name: impl Into<String>) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            entrypoint: CpuKernelEntrypoint::new(),
        }
    }

    pub fn with_entrypoint(mut self, entrypoint: CpuKernelEntrypoint) -> Self {
        self.entrypoint = entrypoint;
        self
    }

    pub fn kernel_name(&self) -> &str {
        &self.kernel_name
    }

    pub fn entrypoint(&self) -> &CpuKernelEntrypoint {
        &self.entrypoint
    }
}

pub type CpuKernelFn =
    fn(&CpuKernelArgs, &CpuKernelLaunchConfig) -> Result<(), CpuKernelLaunchError>;

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
            launch_config: CpuKernelLaunchConfig::new(self.kernel_name),
            kernel_fn: self.kernel_fn,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuKernelLauncher {
    launch_config: CpuKernelLaunchConfig,
    kernel_fn: CpuKernelFn,
}

impl CpuKernelLauncher {
    pub fn kernel_name(&self) -> &str {
        self.launch_config.kernel_name()
    }

    pub fn launch_config(&self) -> &CpuKernelLaunchConfig {
        &self.launch_config
    }

    pub fn entrypoint(&self) -> &CpuKernelEntrypoint {
        self.launch_config.entrypoint()
    }

    pub fn with_entrypoint(mut self, entrypoint: CpuKernelEntrypoint) -> Self {
        self.launch_config = self.launch_config.with_entrypoint(entrypoint);
        self
    }

    pub fn kernel_fn(&self) -> CpuKernelFn {
        self.kernel_fn
    }

    pub fn to_metadata(&self) -> CpuKernelMetadata {
        CpuKernelMetadata::new(self.kernel_name().to_string(), self.kernel_fn)
    }

    pub fn launch(&self, args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
        (self.kernel_fn)(args, &self.launch_config)
    }
}
