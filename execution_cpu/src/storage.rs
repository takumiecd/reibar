use std::sync::{Arc, RwLock};

use schema::DType;

use crate::CpuStorageContext;

#[derive(Debug, Clone)]
pub struct CpuBuffer {
    data: Arc<RwLock<Vec<u8>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuStorageAllocError {
    InvalidByteSizeForDType {
        bytes: usize,
        dtype: DType,
        dtype_size: usize,
    },
    InvalidAlignment {
        alignment: usize,
    },
    UnsupportedAlignment {
        requested: usize,
        supported: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CpuStorage {
    buffer: CpuBuffer,
    dtype: DType,
}

impl CpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
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

impl CpuStorage {
    pub fn new(buffer: CpuBuffer, dtype: DType) -> Self {
        Self { buffer, dtype }
    }

    pub fn allocate(
        _context: &CpuStorageContext,
        bytes: usize,
        alignment: Option<usize>,
        dtype: DType,
    ) -> Result<Self, CpuStorageAllocError> {
        let dtype_size = dtype.size_bytes();
        if bytes % dtype_size != 0 {
            return Err(CpuStorageAllocError::InvalidByteSizeForDType {
                bytes,
                dtype,
                dtype_size,
            });
        }

        if let Some(requested) = alignment {
            if requested == 0 || !requested.is_power_of_two() {
                return Err(CpuStorageAllocError::InvalidAlignment {
                    alignment: requested,
                });
            }
            // Vec<u8> only guarantees alignment of 1 byte.
            if requested > std::mem::align_of::<u8>() {
                return Err(CpuStorageAllocError::UnsupportedAlignment {
                    requested,
                    supported: std::mem::align_of::<u8>(),
                });
            }
        }

        Ok(Self::new(CpuBuffer::new(vec![0u8; bytes]), dtype))
    }

    pub fn len_bytes(&self) -> usize {
        self.buffer.len_bytes()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn buffer(&self) -> &CpuBuffer {
        &self.buffer
    }
}
