use execution::{ExecutionTag, KernelArgs};

use crate::{KernelKey, KeyVersion, OpTag};

pub trait KeyCodec {
    const VERSION: KeyVersion;
    type Parts;
    type Error;

    fn encode(parts: Self::Parts) -> KernelKey;
    fn decode(key: KernelKey) -> Result<Self::Parts, Self::Error>;
}

pub trait KeyResolver {
    fn version(&self) -> KeyVersion;
    fn resolve(&self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey;
}

pub trait KeyFallback {
    type Error;

    fn next(current: KernelKey, args: &KernelArgs) -> Result<Option<KernelKey>, Self::Error>;
}
