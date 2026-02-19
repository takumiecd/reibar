mod cpu;
mod storage;
mod tag;

pub use cpu::CpuExecution;
pub use storage::{CpuStorage, Storage};
pub use tag::{Execution, ExecutionTag};

#[cfg(test)]
mod tests {
    use super::{CpuStorage, Execution, ExecutionTag, Storage};

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
}
