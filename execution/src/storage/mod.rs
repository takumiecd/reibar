use crate::ExecutionTag;
use crate::backend::{BackendBundle, for_each_backend};

macro_rules! define_storage_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq)]
        pub enum Storage {
            $($variant(<$bundle as BackendBundle>::Storage)),+
        }

        impl Storage {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn len(&self) -> usize {
                match self {
                    $(Self::$variant(storage) => <$bundle as BackendBundle>::storage_len(storage)),+
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
