macro_rules! for_each_backend {
    ($macro:ident) => {
        $macro!(Cpu => execution_cpu::CpuBundle);
    };
}

mod capability;
mod kernel;
mod storage;
mod tag;

pub use capability::Capability;
pub use execution_contracts::{
    BackendBundle, ExecutionCapability, ExecutionKernelArgs, ExecutionKernelContext,
    ExecutionKernelLauncher, ExecutionKernelMetadata, ExecutionStorage, ExecutionStorageContext,
};
pub use execution_cpu::{
    CpuBuffer, CpuBundle, CpuCapability, CpuExecution, CpuKernelArgs, CpuKernelContext,
    CpuKernelEntrypoint, CpuKernelFn, CpuKernelLaunchConfig, CpuKernelLaunchError,
    CpuKernelLauncher, CpuKernelMetadata, CpuStorage, CpuStorageAllocError, CpuStorageContext,
};
pub use kernel::{KernelArgs, KernelContext, KernelLaunchError, KernelLauncher, KernelMetadata};
pub use storage::{Storage, StorageAllocError, StorageContext, StorageRequest};
pub use tag::{Execution, ExecutionTag};
