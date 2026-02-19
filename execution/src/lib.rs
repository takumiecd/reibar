mod cpu;
mod kernel;
mod op;
mod storage;
mod tag;

pub use cpu::CpuExecution;
pub use kernel::{
    CpuKernelArgs, CpuKernelContext, CpuKernelFn, CpuKernelLauncher, CpuKernelMetadata, KernelArgs,
    KernelContext, KernelKey, KernelLaunchError, KernelLauncher, KernelMetadata, KernelRegistry,
    KernelRegistryConfig, KernelRegistryError, KernelRegistryStats,
};
pub use op::OpTag;
pub use storage::{CpuStorage, Storage};
pub use tag::{Execution, ExecutionTag};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use schema::{ArgKey, ArgRole, KernelArg};

    use super::{
        CpuKernelArgs, CpuKernelContext, CpuStorage, Execution, ExecutionTag, KernelContext,
        KernelKey, KernelLaunchError, KernelLauncher, KernelMetadata, KernelRegistry,
        KernelRegistryConfig, OpTag, Storage,
    };

    static KERNEL_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel(args: &CpuKernelArgs) -> Result<(), KernelLaunchError> {
        if args.len() == 0 {
            return Ok(());
        }
        KERNEL_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn execution_tag_builds_cpu_execution() {
        let execution = ExecutionTag::Cpu.execution();
        assert_eq!(execution.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn storage_from_tag_is_cpu_storage() {
        let storage = Storage::Cpu(CpuStorage::filled(3, 2.5));
        assert_eq!(storage.tag(), ExecutionTag::Cpu);
        assert_eq!(storage.len(), 3);
        match &storage {
            Storage::Cpu(cpu_storage) => {
                assert_eq!(cpu_storage.as_slice(), &[2.5, 2.5, 2.5]);
            }
        }

        let execution = Execution::cpu();
        assert_eq!(execution.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn kernel_metadata_builds_kernel_launcher() {
        KERNEL_CALL_COUNT.store(0, Ordering::SeqCst);

        let metadata = KernelMetadata::cpu("fill_f32", test_fill_kernel);
        assert_eq!(metadata.tag(), ExecutionTag::Cpu);

        let launcher = metadata.into_launcher();
        assert_eq!(launcher.tag(), ExecutionTag::Cpu);
        let cpu_context = CpuKernelContext::new().with_worker_threads(4);
        let mut args = CpuKernelArgs::new().with_context(cpu_context.clone());
        let input_key = ArgKey::new(ArgRole::Input, "dst");
        let alpha_key = ArgKey::new(ArgRole::Param, "alpha");
        let beta_key = ArgKey::new(ArgRole::Param, "beta");

        args.insert(KernelArg::storage(input_key.clone(), ()))
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
                        .require_f32(&alpha_key)
                        .expect("alpha must exist as f32"),
                    1.0
                );
                assert!(args.require_storage(&input_key).is_ok());
                cpu.launch(&args)
                    .expect("cpu kernel launcher should accept cpu kernel args");
                assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 1);
            }
        }
    }

    #[test]
    fn kernel_context_tag_is_execution_specific() {
        let context = KernelContext::cpu();
        assert_eq!(context.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn kernel_registry_uses_three_tiers() {
        let mut registry = KernelRegistry::new(KernelRegistryConfig {
            cache_capacity: 1,
            main_memory_capacity: 1,
        });

        let fill_key = KernelKey::cpu(OpTag::Fill);
        let copy_key = KernelKey::cpu(OpTag::Copy);

        registry
            .register(fill_key, KernelMetadata::cpu("fill_f32", test_fill_kernel))
            .expect("fill kernel registration should succeed");
        registry
            .register(copy_key, KernelMetadata::cpu("copy_f32", test_fill_kernel))
            .expect("copy kernel registration should succeed");

        let first = registry
            .select(&fill_key)
            .expect("fill kernel should resolve from secondary storage");
        match first {
            KernelLauncher::Cpu(cpu) => assert_eq!(cpu.kernel_name(), "fill_f32"),
        }
        assert_eq!(registry.stats().secondary_hits, 1);

        let second = registry
            .select(&fill_key)
            .expect("fill kernel should resolve from cache");
        match second {
            KernelLauncher::Cpu(cpu) => assert_eq!(cpu.kernel_name(), "fill_f32"),
        }
        assert_eq!(registry.stats().cache_hits, 1);

        let third = registry
            .select(&copy_key)
            .expect("copy kernel should resolve from secondary storage");
        match third {
            KernelLauncher::Cpu(cpu) => assert_eq!(cpu.kernel_name(), "copy_f32"),
        }
        assert_eq!(registry.stats().secondary_hits, 2);

        let fourth = registry
            .select(&fill_key)
            .expect("fill kernel should resolve from secondary after eviction");
        match fourth {
            KernelLauncher::Cpu(cpu) => assert_eq!(cpu.kernel_name(), "fill_f32"),
        }
        assert_eq!(registry.stats().secondary_hits, 3);
    }
}
