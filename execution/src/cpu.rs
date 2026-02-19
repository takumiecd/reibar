use crate::Storage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuExecution;

impl CpuExecution {
    pub fn storage_from_vec(self, data: Vec<f32>) -> Storage {
        Storage::cpu_from_vec(data)
    }

    pub fn storage_filled(self, len: usize, value: f32) -> Storage {
        Storage::cpu_filled(len, value)
    }
}
