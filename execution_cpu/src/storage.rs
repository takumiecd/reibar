use std::sync::{Arc, RwLock};

use crate::CpuStorageContext;

#[derive(Debug, Clone)]
pub struct CpuStorage {
    data: Arc<RwLock<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuStorageAllocError {
    InvalidAlignment {
        alignment: usize,
    },
}

impl CpuStorage {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn allocate(
        _context: &CpuStorageContext,
        bytes: usize,
        alignment: Option<usize>,
    ) -> Result<Self, CpuStorageAllocError> {
        if let Some(requested) = alignment {
            if requested == 0 || !requested.is_power_of_two() {
                return Err(CpuStorageAllocError::InvalidAlignment {
                    alignment: requested,
                });
            }
        }

        Ok(Self::new(vec![0u8; bytes]))
    }

    pub fn len_bytes(&self) -> usize {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.len()
    }

    pub fn with_read_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&data)
    }

    pub fn with_write_bytes<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let mut data = self.data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut data)
    }

    pub fn with_read_ptr<R>(&self, f: impl FnOnce(*const u8, usize) -> R) -> R {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.as_ptr(), data.len())
    }

    pub fn with_write_ptr<R>(&self, f: impl FnOnce(*mut u8, usize) -> R) -> R {
        let mut data = self.data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.as_mut_ptr(), data.len())
    }
}
