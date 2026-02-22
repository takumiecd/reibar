use schema::DType;

use crate::{
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
        dtype: DType,
    ) -> Result<Self::Storage, Self::StorageAllocError>;
    fn storage_len_bytes(storage: &Self::Storage) -> usize;
    fn metadata_into_launcher(metadata: Self::KernelMetadata) -> Self::KernelLauncher;
    fn launcher_to_metadata(launcher: &Self::KernelLauncher) -> Self::KernelMetadata;
    fn launcher_launch(
        launcher: &Self::KernelLauncher,
        args: &Self::KernelArgs,
    ) -> Result<(), Self::LaunchError>;
}
