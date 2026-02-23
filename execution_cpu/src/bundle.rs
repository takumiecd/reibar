use execution_contracts::BackendBundle;
use schema::DType;

use crate::{
    CpuCapability, CpuExecution, CpuKernelArgs, CpuKernelContext, CpuKernelLaunchError,
    CpuKernelLauncher, CpuKernelMetadata, CpuStorage, CpuStorageAllocError, CpuStorageContext,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CpuBundle;

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
        dtype: DType,
    ) -> Result<Self::Storage, Self::StorageAllocError> {
        CpuStorage::allocate(context, bytes, alignment, dtype)
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
