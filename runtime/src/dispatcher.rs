use execution::{ExecutionTag, KernelArgs, KernelLaunchError, KernelMetadata};

use crate::{KernelKey, KernelRegistry, KernelRegistryConfig, KernelRegistryError, OpTag};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    Registry(KernelRegistryError),
    KernelNotFound {
        key: KernelKey,
    },
    ArgsExecutionMismatch {
        key: ExecutionTag,
        args: ExecutionTag,
    },
    Launch(KernelLaunchError),
}

#[derive(Debug, Clone, Default)]
pub struct Dispatcher {
    registry: KernelRegistry,
}

impl Dispatcher {
    pub fn new(config: KernelRegistryConfig) -> Self {
        Self {
            registry: KernelRegistry::new(config),
        }
    }

    pub fn register(
        &mut self,
        key: KernelKey,
        metadata: KernelMetadata,
    ) -> Result<(), DispatchError> {
        self.registry
            .register(key, metadata)
            .map_err(DispatchError::Registry)
    }

    pub fn dispatch(&mut self, key: KernelKey, args: &KernelArgs) -> Result<(), DispatchError> {
        if key.execution() != args.tag() {
            return Err(DispatchError::ArgsExecutionMismatch {
                key: key.execution(),
                args: args.tag(),
            });
        }

        let launcher = self
            .registry
            .select(&key)
            .ok_or(DispatchError::KernelNotFound { key })?;

        launcher.launch(args).map_err(DispatchError::Launch)
    }

    pub fn dispatch_op(
        &mut self,
        execution: ExecutionTag,
        op: OpTag,
        args: &KernelArgs,
    ) -> Result<(), DispatchError> {
        self.dispatch(KernelKey::new(execution, op), args)
    }

    pub fn registry(&self) -> &KernelRegistry {
        &self.registry
    }

    pub fn registry_mut(&mut self) -> &mut KernelRegistry {
        &mut self.registry
    }
}
