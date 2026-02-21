use execution_cpu::{
    CpuBundle, CpuCapability, CpuExecution, CpuKernelArgs, CpuKernelContext, CpuKernelLaunchError,
    CpuKernelLauncher, CpuKernelMetadata, CpuStorage, CpuStorageAllocError, CpuStorageContext,
};

use crate::contracts::{
    ExecutionCapability, ExecutionKernelArgs, ExecutionKernelContext, ExecutionKernelLauncher,
    ExecutionKernelMetadata, ExecutionStorage, ExecutionStorageContext,
};

pub trait BackendBundle {
    type Execution;
    type Storage: ExecutionStorage;
    type StorageContext: ExecutionStorageContext + Default + PartialEq + Eq;
    type StorageAllocError: Clone + PartialEq + Eq;
    type KernelMetadata: ExecutionKernelMetadata<Launcher = Self::KernelLauncher>;
    type KernelLauncher: ExecutionKernelLauncher<
        Metadata = Self::KernelMetadata,
        Args = Self::KernelArgs,
        Error = Self::LaunchError,
    >;
    type KernelContext: ExecutionKernelContext + Default + PartialEq + Eq;
    type KernelArgs: ExecutionKernelArgs;
    type Capability: ExecutionCapability + Default + PartialEq + Eq;
    type LaunchError: Clone + PartialEq + Eq;

    fn execution() -> Self::Execution;
    fn default_context() -> Self::KernelContext;
    fn default_storage_context() -> Self::StorageContext;
    fn default_capability() -> Self::Capability;
    fn allocate_storage(
        context: &Self::StorageContext,
        bytes: usize,
        alignment: Option<usize>,
    ) -> Result<Self::Storage, Self::StorageAllocError>;
    fn storage_len_bytes(storage: &Self::Storage) -> usize;
    fn metadata_into_launcher(metadata: Self::KernelMetadata) -> Self::KernelLauncher;
    fn launcher_to_metadata(launcher: &Self::KernelLauncher) -> Self::KernelMetadata;
    fn launcher_launch(
        launcher: &Self::KernelLauncher,
        args: &Self::KernelArgs,
    ) -> Result<(), Self::LaunchError>;
}

impl ExecutionStorage for CpuStorage {
    fn len_bytes(&self) -> usize {
        self.len_bytes()
    }
}

impl ExecutionStorageContext for CpuStorageContext {}

impl ExecutionKernelContext for CpuKernelContext {}

impl ExecutionKernelArgs for CpuKernelArgs {}

impl ExecutionCapability for CpuCapability {}

impl ExecutionKernelMetadata for CpuKernelMetadata {
    type Launcher = CpuKernelLauncher;

    fn into_launcher(self) -> Self::Launcher {
        self.into_launcher()
    }
}

impl ExecutionKernelLauncher for CpuKernelLauncher {
    type Metadata = CpuKernelMetadata;
    type Args = CpuKernelArgs;
    type Error = CpuKernelLaunchError;

    fn to_metadata(&self) -> Self::Metadata {
        self.to_metadata()
    }

    fn launch(&self, args: &Self::Args) -> Result<(), Self::Error> {
        self.launch(args)
    }
}

impl BackendBundle for CpuBundle {
    type Execution = CpuExecution;
    type Storage = CpuStorage;
    type StorageContext = CpuStorageContext;
    type StorageAllocError = CpuStorageAllocError;
    type KernelMetadata = CpuKernelMetadata;
    type KernelLauncher = CpuKernelLauncher;
    type KernelContext = CpuKernelContext;
    type KernelArgs = CpuKernelArgs;
    type Capability = CpuCapability;
    type LaunchError = CpuKernelLaunchError;

    fn execution() -> Self::Execution {
        CpuExecution
    }

    fn default_context() -> Self::KernelContext {
        CpuKernelContext::default()
    }

    fn default_storage_context() -> Self::StorageContext {
        CpuStorageContext::default()
    }

    fn default_capability() -> Self::Capability {
        CpuCapability::default()
    }

    fn allocate_storage(
        context: &Self::StorageContext,
        bytes: usize,
        alignment: Option<usize>,
    ) -> Result<Self::Storage, Self::StorageAllocError> {
        CpuStorage::allocate(context, bytes, alignment)
    }

    fn storage_len_bytes(storage: &Self::Storage) -> usize {
        storage.len_bytes()
    }

    fn metadata_into_launcher(metadata: Self::KernelMetadata) -> Self::KernelLauncher {
        metadata.into_launcher()
    }

    fn launcher_to_metadata(launcher: &Self::KernelLauncher) -> Self::KernelMetadata {
        launcher.to_metadata()
    }

    fn launcher_launch(
        launcher: &Self::KernelLauncher,
        args: &Self::KernelArgs,
    ) -> Result<(), Self::LaunchError> {
        launcher.launch(args)
    }
}

macro_rules! for_each_backend {
    ($macro:ident) => {
        $macro!(Cpu => execution_cpu::CpuBundle);
    };
}

pub(crate) use for_each_backend;
