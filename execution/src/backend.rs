use execution_cpu::{
    CpuBundle, CpuCapability, CpuExecution, CpuKernelArgs, CpuKernelContext, CpuKernelLaunchError,
    CpuKernelLauncher, CpuKernelMetadata, CpuStorage,
};

pub trait BackendBundle {
    type Execution;
    type Storage;
    type KernelMetadata;
    type KernelLauncher;
    type KernelContext;
    type KernelArgs;
    type Capability;
    type LaunchError;

    fn execution() -> Self::Execution;
    fn default_context() -> Self::KernelContext;
    fn default_capability() -> Self::Capability;
    fn storage_len(storage: &Self::Storage) -> usize;
    fn metadata_into_launcher(metadata: Self::KernelMetadata) -> Self::KernelLauncher;
    fn launcher_to_metadata(launcher: &Self::KernelLauncher) -> Self::KernelMetadata;
    fn launcher_launch(
        launcher: &Self::KernelLauncher,
        args: &Self::KernelArgs,
    ) -> Result<(), Self::LaunchError>;
}

impl BackendBundle for CpuBundle {
    type Execution = CpuExecution;
    type Storage = CpuStorage;
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

    fn default_capability() -> Self::Capability {
        CpuCapability::default()
    }

    fn storage_len(storage: &Self::Storage) -> usize {
        storage.len()
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
