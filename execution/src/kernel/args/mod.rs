use crate::ExecutionTag;
use crate::backend::{BackendBundle, for_each_backend};

macro_rules! define_kernel_args_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, PartialEq)]
        pub enum KernelArgs {
            $($variant(<$bundle as BackendBundle>::KernelArgs)),+
        }

        impl KernelArgs {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }
        }

        $(
        impl From<<$bundle as BackendBundle>::KernelArgs> for KernelArgs {
            fn from(value: <$bundle as BackendBundle>::KernelArgs) -> Self {
                Self::$variant(value)
            }
        }
        )+
    };
}

for_each_backend!(define_kernel_args_types);

impl KernelArgs {
    pub fn cpu(args: execution_cpu::CpuKernelArgs) -> Self {
        Self::Cpu(args)
    }
}
