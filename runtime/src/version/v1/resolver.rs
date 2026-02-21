use execution::{ExecutionTag, KernelArgs};

use crate::version::api::KeyResolver;
use crate::version::v1::codec::{V1KeyCodec, V1KeyParts};
use crate::{KernelKey, KeyVersion, OpTag};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V1KeyResolver;

impl KeyResolver for V1KeyResolver {
    fn version(&self) -> KeyVersion {
        KeyVersion::V1
    }

    fn resolve(&self, execution: ExecutionTag, op: OpTag, _args: &KernelArgs) -> KernelKey {
        V1KeyCodec::encode(V1KeyParts {
            execution,
            op,
            selector: 0,
        })
    }
}
