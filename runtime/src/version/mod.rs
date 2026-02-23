mod api;
mod v1;
mod v2;

use execution::{ExecutionTag, KernelArgs};

use crate::{KernelKey, KeyVersion, OpTag};

pub use api::{KeyCodec, KeyFallback, KeyResolver};
pub use v1::{V1Fallback, V1KeyCodec, V1KeyError, V1KeyParts, V1KeyResolver};
pub use v2::{V2Fallback, V2KeyCodec, V2KeyError, V2KeyParts, V2KeyResolver};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolverKind {
    #[default]
    V1,
    V2,
}

impl ResolverKind {
    pub fn version(self) -> KeyVersion {
        match self {
            Self::V1 => KeyVersion::V1,
            Self::V2 => KeyVersion::V2,
        }
    }

    pub fn resolve(self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey {
        match self {
            Self::V1 => V1KeyResolver.resolve(execution, op, args),
            Self::V2 => V2KeyResolver.resolve(execution, op, args),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionError {
    V1(V1KeyError),
    V2(V2KeyError),
}

pub fn next_fallback_key(
    version: KeyVersion,
    current: KernelKey,
    args: &KernelArgs,
) -> Result<Option<KernelKey>, VersionError> {
    match version {
        KeyVersion::V1 => v1::V1Fallback::next(current, args).map_err(VersionError::V1),
        KeyVersion::V2 => v2::V2Fallback::next(current, args).map_err(VersionError::V2),
    }
}
