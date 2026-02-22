use execution_contracts::BackendBundle;

macro_rules! define_execution_types {
    ($($variant:ident => $bundle:path),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum ExecutionTag {
            $($variant),+
        }

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub enum Execution {
            $($variant(<$bundle as BackendBundle>::Execution)),+
        }

        impl Execution {
            pub fn tag(&self) -> ExecutionTag {
                match self {
                    $(Self::$variant(_) => ExecutionTag::$variant),+
                }
            }
        }

        impl ExecutionTag {
            pub fn execution(self) -> Execution {
                match self {
                    $(Self::$variant => Execution::$variant(<$bundle as BackendBundle>::execution())),+
                }
            }
        }
    };
}

for_each_backend!(define_execution_types);
