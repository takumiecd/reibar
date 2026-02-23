use execution::KernelArgs;

use crate::KernelKey;
use crate::version::api::KeyFallback;
use crate::version::v1::codec::{V1KeyCodec, V1KeyError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V1Fallback;

impl KeyFallback for V1Fallback {
    type Error = V1KeyError;

    fn next(current: KernelKey, _args: &KernelArgs) -> Result<Option<KernelKey>, Self::Error> {
        let mut parts = V1KeyCodec::decode(current)?;
        if parts.selector == 0 {
            return Ok(None);
        }
        parts.selector = parts.selector.saturating_sub(1);
        Ok(Some(V1KeyCodec::encode(parts)))
    }
}
