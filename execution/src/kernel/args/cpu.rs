use schema::{ArgKey, KernelArg, KernelArgs, KernelArgsError, StorageValue};

use super::super::context::CpuKernelContext;

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

    pub fn require_storage(&self, key: &ArgKey) -> Result<&(), KernelArgsError> {
        self.args.require_as::<StorageValue>(key)
    }

    pub fn require_f32(&self, key: &ArgKey) -> Result<&f32, KernelArgsError> {
        self.args.require_as::<f32>(key)
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
