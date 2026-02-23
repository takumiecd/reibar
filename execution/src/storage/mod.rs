mod context;

pub use context::{StorageAllocError, StorageContext, StorageRequest};

use crate::ExecutionTag;
use execution_contracts::BackendBundle;
use schema::DType;

macro_rules! define_storage_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone)]
        pub enum Storage {
            $($variant(<$bundle as BackendBundle>::Storage)),+
        }

        impl Storage {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn len_bytes(&self) -> usize {
                match self {
                    $(Self::$variant(storage) => <$bundle as BackendBundle>::storage_len_bytes(storage)),+
                }
            }

            pub fn dtype(&self) -> DType {
                match self {
                    $(Self::$variant(storage) => storage.dtype()),+
                }
            }

            pub fn allocate(
                tag: ExecutionTag,
                context: &StorageContext,
                request: StorageRequest,
            ) -> Result<Self, StorageAllocError> {
                if tag != context.tag() {
                    return Err(StorageAllocError::ContextTagMismatch {
                        requested: tag,
                        context: context.tag(),
                    });
                }

                match context {
                    $(
                        StorageContext::$variant(storage_context) => {
                            <$bundle as BackendBundle>::allocate_storage(
                                storage_context,
                                request.bytes,
                                request.alignment,
                                request.dtype,
                            )
                            .map(Self::$variant)
                            .map_err(StorageAllocError::$variant)
                        }
                    ),+
                }
            }
        }

    };
}

for_each_backend!(define_storage_types);

#[cfg(test)]
mod tests {
    use schema::DType;

    use super::{Storage, StorageContext, StorageRequest};
    use crate::{CpuBuffer, CpuStorage, ExecutionTag};

    #[test]
    fn storage_from_tag_is_cpu_storage() {
        let storage = Storage::Cpu(
            CpuStorage::new(
                CpuBuffer::new_with_alignment(
                    encode_f32_slice(&[2.5, 2.5, 2.5]),
                    DType::F32.alignment(),
                )
                .expect("aligned buffer creation should succeed"),
                DType::F32,
            )
            .expect("typed storage creation should succeed"),
        );
        assert_eq!(storage.tag(), ExecutionTag::Cpu);
        assert_eq!(storage.len_bytes(), 12);
        assert_eq!(storage.dtype(), DType::F32);
        match &storage {
            Storage::Cpu(cpu_storage) => {
                let values = cpu_storage.buffer().with_read_bytes(decode_f32_vec);
                assert_eq!(values, vec![2.5, 2.5, 2.5]);
            }
        }

        let execution = ExecutionTag::Cpu.execution();
        assert_eq!(execution.tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn storage_can_be_allocated_from_context() {
        let context = StorageContext::from_execution_tag(ExecutionTag::Cpu);
        let storage = Storage::allocate(
            ExecutionTag::Cpu,
            &context,
            StorageRequest::new(16, DType::F32).with_alignment(64),
        )
        .expect("storage allocation should succeed");

        assert_eq!(storage.tag(), ExecutionTag::Cpu);
        assert_eq!(storage.len_bytes(), 16);
        assert_eq!(storage.dtype(), DType::F32);
        match storage {
            Storage::Cpu(cpu_storage) => {
                assert!(cpu_storage.alignment() >= 64);
            }
        }
    }

    // Verifies the ExecutionStorage Clone contract: clone() must be a shared handle,
    // not a deep copy. A write through the clone must be visible in the original.
    #[test]
    fn storage_clone_is_shallow_shared_handle() {
        let context = StorageContext::from_execution_tag(ExecutionTag::Cpu);
        let original = Storage::allocate(
            ExecutionTag::Cpu,
            &context,
            StorageRequest::new(std::mem::size_of::<f32>(), DType::F32),
        )
        .expect("storage allocation should succeed");

        let handle = original.clone();

        let encoded = 42.0f32.to_ne_bytes();
        match &handle {
            Storage::Cpu(cpu) => cpu.buffer().with_write_bytes(|bytes| {
                bytes[0..4].copy_from_slice(&encoded);
            }),
        }

        let value = match &original {
            Storage::Cpu(cpu) => cpu.buffer().with_read_bytes(|bytes| {
                f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            }),
        };

        assert_eq!(
            value, 42.0f32,
            "clone() must share the underlying allocation (shallow copy)"
        );
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
