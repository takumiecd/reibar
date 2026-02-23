mod args;
mod context;

pub use args::KernelArgs;
pub use context::KernelContext;

use execution_contracts::BackendBundle;

use crate::ExecutionTag;

macro_rules! define_kernel_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone)]
        pub enum KernelMetadata {
            $($variant(<$bundle as BackendBundle>::KernelMetadata)),+
        }

        #[derive(Debug, Clone)]
        pub enum KernelLauncher {
            $($variant(<$bundle as BackendBundle>::KernelLauncher)),+
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum KernelLaunchError {
            ExecutionMismatch {
                launcher: ExecutionTag,
                args: ExecutionTag,
            },
            $($variant(<$bundle as BackendBundle>::LaunchError)),+
        }

        impl KernelMetadata {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn into_launcher(self) -> KernelLauncher {
                match self {
                    $(Self::$variant(metadata) => {
                        KernelLauncher::$variant(<$bundle as BackendBundle>::metadata_into_launcher(metadata))
                    }),+
                }
            }
        }

        impl KernelLauncher {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn launch(&self, args: &KernelArgs) -> Result<(), KernelLaunchError> {
                if self.tag() != args.tag() {
                    return Err(KernelLaunchError::ExecutionMismatch {
                        launcher: self.tag(),
                        args: args.tag(),
                    });
                }

                match (self, args) {
                    $(
                        (Self::$variant(launcher), KernelArgs::$variant(backend_args)) => {
                            <$bundle as BackendBundle>::launcher_launch(launcher, backend_args)
                                .map_err(KernelLaunchError::$variant)
                        }
                    ),+
                }
            }

            pub fn to_metadata(&self) -> KernelMetadata {
                match self {
                    $(
                        Self::$variant(launcher) => {
                            KernelMetadata::$variant(<$bundle as BackendBundle>::launcher_to_metadata(launcher))
                        }
                    ),+
                }
            }
        }
    };
}

for_each_backend!(define_kernel_types);

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg, StorageValue};

    use super::{KernelContext, KernelLauncher, KernelMetadata};
    use crate::{
        CpuBuffer, CpuKernelArgs, CpuKernelContext, CpuKernelLaunchConfig, CpuKernelLaunchError,
        CpuKernelMetadata, CpuStorage, ExecutionTag,
    };

    static KERNEL_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel(
        args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        if args.is_empty() {
            return Ok(());
        }
        KERNEL_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn kernel_metadata_builds_kernel_launcher() {
        KERNEL_CALL_COUNT.store(0, Ordering::SeqCst);

        let metadata = KernelMetadata::Cpu(CpuKernelMetadata::new("fill_f32", test_fill_kernel));
        assert_eq!(metadata.tag(), ExecutionTag::Cpu);

        let launcher = metadata.into_launcher();
        assert_eq!(launcher.tag(), ExecutionTag::Cpu);
        let cpu_context = CpuKernelContext::new().with_worker_threads(4);
        let mut args = CpuKernelArgs::new().with_context(cpu_context.clone());
        let input_key = ArgKey::new(ArgRole::Input, "dst", ArgKind::Storage);
        let alpha_key = ArgKey::new(ArgRole::Param, "alpha", ArgKind::F32);
        let beta_key = ArgKey::new(ArgRole::Param, "beta", ArgKind::F32);

        args.insert(KernelArg::storage(
            input_key.clone(),
            CpuStorage::new(
                CpuBuffer::new_with_alignment(vec![0u8; 4], DType::F32.alignment())
                    .expect("aligned buffer creation should succeed"),
                DType::F32,
            )
            .expect("typed storage creation should succeed"),
        ))
        .expect("storage insertion should succeed");
        args.insert(KernelArg::f32(alpha_key.clone(), 1.0))
            .expect("alpha insertion should succeed");
        args.insert(KernelArg::f32(beta_key.clone(), 2.0))
            .expect("beta insertion should succeed");

        match launcher {
            KernelLauncher::Cpu(cpu) => {
                assert_eq!(cpu.kernel_name(), "fill_f32");
                assert_eq!(args.len(), 3);
                assert_eq!(args.context(), &cpu_context);
                assert_eq!(
                    *args
                        .args()
                        .require_as::<f32>(&alpha_key)
                        .expect("alpha must exist as f32"),
                    1.0
                );
                assert!(args.args().require_as::<StorageValue>(&input_key).is_ok());
                cpu.launch(&args)
                    .expect("cpu kernel launcher should accept cpu kernel args");
                assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 1);
            }
        }
    }

    #[test]
    fn kernel_context_tag_is_execution_specific() {
        let context = KernelContext::from_execution_tag(ExecutionTag::Cpu);
        assert_eq!(context.tag(), ExecutionTag::Cpu);
    }
}
