use crate::{ArgKey, ArgKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelArgsError {
    DuplicateKey {
        key: ArgKey,
    },
    MissingKey {
        key: ArgKey,
    },
    TypeMismatch {
        key: ArgKey,
        expected: ArgKind,
        actual: ArgKind,
    },
}
