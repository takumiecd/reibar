use crate::ExecutionTag;
use crate::backend::{BackendBundle, for_each_backend};

macro_rules! define_capability_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum Capability {
            $($variant(<$bundle as BackendBundle>::Capability)),+
        }

        impl Capability {
            pub fn from_execution_tag(tag: ExecutionTag) -> Self {
                match tag {
                    $(ExecutionTag::$variant => Self::$variant(<$bundle as BackendBundle>::default_capability())),+
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

for_each_backend!(define_capability_types);
