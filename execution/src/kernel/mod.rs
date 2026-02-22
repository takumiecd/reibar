mod args;
mod context;

pub use args::KernelArgs;
pub use context::KernelContext;

use execution_contracts::BackendBundle;

use crate::ExecutionTag;

macro_rules! define_kernel_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone)]
        pub enum KernelMetadata {
            $($variant(<$bundle as BackendBundle>::KernelMetadata)),+
        }

        #[derive(Debug, Clone)]
        pub enum KernelLauncher {
            $($variant(<$bundle as BackendBundle>::KernelLauncher)),+
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum KernelLaunchError {
            ExecutionMismatch {
                launcher: ExecutionTag,
                args: ExecutionTag,
            },
            $($variant(<$bundle as BackendBundle>::LaunchError)),+
        }

        impl KernelMetadata {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn into_launcher(self) -> KernelLauncher {
                match self {
                    $(Self::$variant(metadata) => {
                        KernelLauncher::$variant(<$bundle as BackendBundle>::metadata_into_launcher(metadata))
                    }),+
                }
            }
        }

        impl KernelLauncher {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }

            pub fn launch(&self, args: &KernelArgs) -> Result<(), KernelLaunchError> {
                if self.tag() != args.tag() {
                    return Err(KernelLaunchError::ExecutionMismatch {
                        launcher: self.tag(),
                        args: args.tag(),
                    });
                }

                match (self, args) {
                    $(
                        (Self::$variant(launcher), KernelArgs::$variant(backend_args)) => {
                            <$bundle as BackendBundle>::launcher_launch(launcher, backend_args)
                                .map_err(KernelLaunchError::$variant)
                        }
                    ),+
                }
            }

            pub fn to_metadata(&self) -> KernelMetadata {
                match self {
                    $(
                        Self::$variant(launcher) => {
                            KernelMetadata::$variant(<$bundle as BackendBundle>::launcher_to_metadata(launcher))
                        }
                    ),+
                }
            }
        }
    };
}

for_each_backend!(define_kernel_types);
