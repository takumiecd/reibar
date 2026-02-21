use execution::ExecutionTag;

use crate::version::api::KeyCodec;
use crate::{KernelKey, KeyError, KeyVersion, OpTag, compose_key, key_payload, key_version};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V2KeyParts {
    pub op: OpTag,
    pub layout: u8,
    pub execution: ExecutionTag,
    pub fingerprint: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2KeyError {
    Version(KeyError),
    VersionMismatch {
        expected: KeyVersion,
        actual: KeyVersion,
    },
    InvalidExecution(u8),
    InvalidOp(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2KeyCodec;

impl KeyCodec for V2KeyCodec {
    const VERSION: KeyVersion = KeyVersion::V2;
    type Parts = V2KeyParts;
    type Error = V2KeyError;

    // V2 payload layout (intentionally different from V1):
    // [59:52] op
    // [51:44] layout
    // [43:40] execution
    // [39:00] fingerprint
    fn encode(parts: Self::Parts) -> KernelKey {
        let op_bits = u64::from(parts.op.code());
        let layout_bits = u64::from(parts.layout);
        let execution_bits = u64::from(encode_execution(parts.execution) & 0x0F);
        let fingerprint_bits = parts.fingerprint & ((1u64 << 40) - 1);
        let payload =
            (op_bits << 52) | (layout_bits << 44) | (execution_bits << 40) | fingerprint_bits;
        compose_key(Self::VERSION, payload)
    }

    fn decode(key: KernelKey) -> Result<Self::Parts, Self::Error> {
        ensure_version(key, Self::VERSION)?;

        let payload = key_payload(key);
        let op_code = ((payload >> 52) & 0xFF) as u8;
        let layout = ((payload >> 44) & 0xFF) as u8;
        let execution_code = ((payload >> 40) & 0x0F) as u8;
        let fingerprint = payload & ((1u64 << 40) - 1);

        let execution =
            decode_execution(execution_code).ok_or(V2KeyError::InvalidExecution(execution_code))?;
        let op = OpTag::from_code(op_code).ok_or(V2KeyError::InvalidOp(op_code))?;

        Ok(V2KeyParts {
            op,
            layout,
            execution,
            fingerprint,
        })
    }
}

impl V2KeyCodec {
    pub fn encode(parts: V2KeyParts) -> KernelKey {
        <Self as KeyCodec>::encode(parts)
    }

    pub fn decode(key: KernelKey) -> Result<V2KeyParts, V2KeyError> {
        <Self as KeyCodec>::decode(key)
    }
}

fn ensure_version(key: KernelKey, expected: KeyVersion) -> Result<(), V2KeyError> {
    let actual = key_version(key).map_err(V2KeyError::Version)?;
    if actual != expected {
        return Err(V2KeyError::VersionMismatch { expected, actual });
    }
    Ok(())
}

fn encode_execution(tag: ExecutionTag) -> u8 {
    match tag {
        ExecutionTag::Cpu => 0,
    }
}

fn decode_execution(code: u8) -> Option<ExecutionTag> {
    match code {
        0 => Some(ExecutionTag::Cpu),
        _ => None,
    }
}
