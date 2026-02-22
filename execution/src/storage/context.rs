use crate::ExecutionTag;
use execution_contracts::BackendBundle;
use schema::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageRequest {
    pub bytes: usize,
    pub dtype: DType,
    pub alignment: Option<usize>,
}

impl StorageRequest {
    pub fn new(bytes: usize, dtype: DType) -> Self {
        Self {
            bytes,
            dtype,
            alignment: None,
        }
    }

    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = Some(alignment);
        self
    }
}

macro_rules! define_storage_context_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum StorageContext {
            $($variant(<$bundle as BackendBundle>::StorageContext)),+
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum StorageAllocError {
            ContextTagMismatch {
                requested: ExecutionTag,
                context: ExecutionTag,
            },
            $($variant(<$bundle as BackendBundle>::StorageAllocError)),+
        }

        impl StorageContext {
            pub fn from_execution_tag(tag: ExecutionTag) -> Self {
                match tag {
                    $(ExecutionTag::$variant => Self::$variant(<$bundle as BackendBundle>::default_storage_context())),+
                }
            }

            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }
        }
    };
}

for_each_backend!(define_storage_context_types);
