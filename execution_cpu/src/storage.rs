use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct CpuStorage {
    data: Arc<RwLock<Vec<f32>>>,
}

impl CpuStorage {
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn len(&self) -> usize {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.len()
    }

    pub fn with_read<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&data)
    }

    pub fn with_write<R>(&self, f: impl FnOnce(&mut [f32]) -> R) -> R {
        let mut data = self.data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut data)
    }
}
