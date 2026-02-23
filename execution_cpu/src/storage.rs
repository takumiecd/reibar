use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, RwLock};

use schema::DType;

use crate::CpuStorageContext;

#[derive(Debug)]
struct CpuRawBuffer {
    ptr: NonNull<u8>,
    len: usize,
    alignment: usize,
}

// SAFETY: CpuRawBuffer owns a heap allocation and plain metadata only.
// Cross-thread access is synchronized externally via RwLock in CpuBuffer.
unsafe impl Send for CpuRawBuffer {}
// SAFETY: Shared access is read-only unless synchronized mutable access is used.
unsafe impl Sync for CpuRawBuffer {}

impl CpuRawBuffer {
    fn zeroed(len: usize, alignment: usize) -> Result<Self, CpuStorageAllocError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CpuStorageAllocError::InvalidAlignment { alignment });
        }

        let layout = Layout::from_size_align(len, alignment)
            .map_err(|_| CpuStorageAllocError::LayoutError { bytes: len, alignment })?;

        if len == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                alignment,
            });
        }

        // SAFETY: layout is valid and non-zero in size here.
        let ptr = unsafe { alloc_zeroed(layout) };
        let Some(ptr) = NonNull::new(ptr) else {
            return Err(CpuStorageAllocError::AllocationFailed {
                bytes: len,
                alignment,
            });
        };

        Ok(Self {
            ptr,
            len,
            alignment,
        })
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr points to a valid allocation for len bytes (or dangling for len=0).
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr points to a unique valid allocation for len bytes (or dangling for len=0).
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for CpuRawBuffer {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        if let Ok(layout) = Layout::from_size_align(self.len, self.alignment) {
            // SAFETY: ptr was allocated with this exact layout.
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

#[derive(Debug, Clone)]
pub struct CpuBuffer {
    data: Arc<RwLock<CpuRawBuffer>>,
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
    LayoutError {
        bytes: usize,
        alignment: usize,
    },
    AllocationFailed {
        bytes: usize,
        alignment: usize,
    },
    BufferAlignmentTooSmall {
        buffer_alignment: usize,
        required_alignment: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CpuStorage {
    buffer: CpuBuffer,
    dtype: DType,
}

impl CpuBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        // Alignment 1 is always valid for raw bytes.
        Self::new_with_alignment(data, 1).expect("byte buffer alignment=1 must be valid")
    }

    pub fn new_with_alignment(
        data: Vec<u8>,
        alignment: usize,
    ) -> Result<Self, CpuStorageAllocError> {
        let mut raw = CpuRawBuffer::zeroed(data.len(), alignment)?;
        raw.as_mut_slice().copy_from_slice(&data);
        Ok(Self {
            data: Arc::new(RwLock::new(raw)),
        })
    }

    pub fn allocate_zeroed(bytes: usize, alignment: usize) -> Result<Self, CpuStorageAllocError> {
        Ok(Self {
            data: Arc::new(RwLock::new(CpuRawBuffer::zeroed(bytes, alignment)?)),
        })
    }

    pub fn len_bytes(&self) -> usize {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.len
    }

    pub fn alignment(&self) -> usize {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.alignment
    }

    pub fn with_read_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.as_slice())
    }

    pub fn with_write_bytes<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        let mut data = self.data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.as_mut_slice())
    }

    pub fn with_read_ptr<R>(&self, f: impl FnOnce(*const u8, usize) -> R) -> R {
        let data = self.data.read().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.ptr.as_ptr(), data.len)
    }

    pub fn with_write_ptr<R>(&self, f: impl FnOnce(*mut u8, usize) -> R) -> R {
        let data = self.data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        f(data.ptr.as_ptr(), data.len)
    }
}

impl CpuStorage {
    pub fn new(buffer: CpuBuffer, dtype: DType) -> Result<Self, CpuStorageAllocError> {
        let required_alignment = dtype.alignment();
        let buffer_alignment = buffer.alignment();
        if buffer_alignment < required_alignment {
            return Err(CpuStorageAllocError::BufferAlignmentTooSmall {
                buffer_alignment,
                required_alignment,
            });
        }
        Ok(Self { buffer, dtype })
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

        let requested_alignment = alignment.unwrap_or(dtype.alignment());
        let effective_alignment = requested_alignment.max(dtype.alignment());
        let buffer = CpuBuffer::allocate_zeroed(bytes, effective_alignment)?;
        Self::new(buffer, dtype)
    }

    pub fn len_bytes(&self) -> usize {
        self.buffer.len_bytes()
    }

    pub fn alignment(&self) -> usize {
        self.buffer.alignment()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn buffer(&self) -> &CpuBuffer {
        &self.buffer
    }
}

impl execution_contracts::ExecutionStorage for CpuStorage {
    fn len_bytes(&self) -> usize {
        self.len_bytes()
    }

    fn dtype(&self) -> DType {
        self.dtype()
    }
}
