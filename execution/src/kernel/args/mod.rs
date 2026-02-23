use crate::ExecutionTag;
use execution_contracts::BackendBundle;

macro_rules! define_kernel_args_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone)]
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

    };
}

for_each_backend!(define_kernel_args_types);
