mod cpu;
mod kernel;
mod storage;
mod tag;

pub use cpu::CpuExecution;
pub use kernel::{
    CpuKernelArgs, CpuKernelLauncher, CpuKernelMetadata, KernelArgs, KernelLaunchError,
    KernelLauncher, KernelMetadata,
};
pub use storage::{CpuStorage, Storage};
pub use tag::{Execution, ExecutionTag};

#[cfg(test)]
mod tests {
    use super::{
        CpuKernelArgs, CpuStorage, Execution, ExecutionTag, KernelLauncher, KernelMetadata, Storage,
    };

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
        let metadata = KernelMetadata::cpu("fill_f32");
        assert_eq!(metadata.tag(), ExecutionTag::Cpu);

        let launcher = metadata.into_launcher();
        assert_eq!(launcher.tag(), ExecutionTag::Cpu);
        let args = CpuKernelArgs::new()
            .with_storage_count(1)
            .with_param_count(2);

        match launcher {
            KernelLauncher::Cpu(cpu) => {
                assert_eq!(cpu.kernel_name(), "fill_f32");
                assert_eq!(args.storage_count(), 1);
                assert_eq!(args.param_count(), 2);
                cpu.launch(&args)
                    .expect("cpu kernel launcher should accept cpu kernel args");
            }
        }
    }
}
