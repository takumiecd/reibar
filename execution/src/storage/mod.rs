mod context;

pub use context::{StorageAllocError, StorageContext, StorageRequest};

use crate::ExecutionTag;
use crate::backend::{BackendBundle, for_each_backend};
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

        $(
        impl From<<$bundle as BackendBundle>::Storage> for Storage {
            fn from(value: <$bundle as BackendBundle>::Storage) -> Self {
                Storage::$variant(value)
            }
        }
        )+
    };
}

for_each_backend!(define_storage_types);
