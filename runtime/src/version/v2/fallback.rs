use execution::KernelArgs;

use crate::KernelKey;
use crate::version::api::KeyFallback;
use crate::version::v2::codec::{V2KeyCodec, V2KeyError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2Fallback;

impl KeyFallback for V2Fallback {
    type Error = V2KeyError;

    fn next(current: KernelKey, _args: &KernelArgs) -> Result<Option<KernelKey>, Self::Error> {
        let mut parts = V2KeyCodec::decode(current)?;

        if parts.layout > 0 {
            parts.layout = parts.layout.saturating_sub(1);
            return Ok(Some(V2KeyCodec::encode(parts)));
        }

        if parts.fingerprint != 0 {
            parts.fingerprint = 0;
            return Ok(Some(V2KeyCodec::encode(parts)));
        }

        Ok(None)
    }
}
