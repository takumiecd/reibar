use crate::ExecutionTag;
use crate::backend::{BackendBundle, for_each_backend};

macro_rules! define_kernel_context_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum KernelContext {
            $($variant(<$bundle as BackendBundle>::KernelContext)),+
        }

        impl KernelContext {
            pub fn from_execution_tag(tag: ExecutionTag) -> Self {
                match tag {
                    $(ExecutionTag::$variant => Self::$variant(<$bundle as BackendBundle>::default_context())),+
                }
            }

            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }
        }

        $(
        impl From<<$bundle as BackendBundle>::KernelContext> for KernelContext {
            fn from(value: <$bundle as BackendBundle>::KernelContext) -> Self {
                Self::$variant(value)
            }
        }
        )+
    };
}

for_each_backend!(define_kernel_context_types);

impl KernelContext {
    pub fn cpu() -> Self {
        Self::Cpu(execution_cpu::CpuKernelContext::new())
    }
}
