mod backend;
mod capability;
mod contracts;
mod kernel;
mod storage;
mod tag;

pub use capability::Capability;
pub use execution_cpu::{
    CpuBundle, CpuCapability, CpuExecution, CpuKernelArgs, CpuKernelContext, CpuKernelFn,
    CpuKernelLaunchError, CpuKernelLauncher, CpuKernelMetadata, CpuStorage, CpuStorageAllocError,
    CpuStorageContext,
};
pub use contracts::*;
pub use kernel::{KernelArgs, KernelContext, KernelLaunchError, KernelLauncher, KernelMetadata};
pub use storage::{Storage, StorageAllocError, StorageContext, StorageRequest};
pub use tag::{Execution, ExecutionTag};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use schema::{ArgKey, ArgKind, ArgRole, KernelArg, StorageValue};

    use super::{
        Capability, CpuKernelArgs, CpuKernelContext, CpuKernelLaunchError, CpuKernelMetadata,
        CpuStorage, Execution, ExecutionTag, KernelContext, KernelLauncher, KernelMetadata,
        Storage, StorageContext, StorageRequest,
    };

    static KERNEL_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel(args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
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
        let storage = Storage::Cpu(CpuStorage::new(encode_f32_slice(&[2.5, 2.5, 2.5])));
        assert_eq!(storage.tag(), ExecutionTag::Cpu);
        assert_eq!(storage.len_bytes(), 12);
        match &storage {
            Storage::Cpu(cpu_storage) => {
                let values = cpu_storage.with_read_bytes(decode_f32_vec);
                assert_eq!(values, vec![2.5, 2.5, 2.5]);
            }
        }

        let execution = Execution::cpu();
        assert_eq!(execution.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn storage_can_be_allocated_from_context() {
        let context = StorageContext::cpu();
        let storage = Storage::allocate(
            ExecutionTag::Cpu,
            &context,
            StorageRequest::new(16).with_alignment(4),
        )
        .expect("storage allocation should succeed");

        assert_eq!(storage.tag(), ExecutionTag::Cpu);
        assert_eq!(storage.len_bytes(), 16);
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
            CpuStorage::new(vec![0u8; 4]),
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
        let context = KernelContext::cpu();
        assert_eq!(context.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn capability_follows_execution_tag() {
        let capability = Capability::from_execution_tag(ExecutionTag::Cpu);
        assert_eq!(capability.tag(), ExecutionTag::Cpu);
    }

    fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
        let mut chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
        let values: Vec<f32> = chunks
            .by_ref()
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert!(chunks.remainder().is_empty());
        values
    }
}
