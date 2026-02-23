use execution::ExecutionTag;

use crate::version::api::KeyCodec;
use crate::{KernelKey, KeyError, KeyVersion, OpTag, compose_key, key_payload, key_version};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V1KeyParts {
    pub execution: ExecutionTag,
    pub op: OpTag,
    pub selector: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V1KeyError {
    Version(KeyError),
    VersionMismatch {
        expected: KeyVersion,
        actual: KeyVersion,
    },
    InvalidExecution(u8),
    InvalidOp(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V1KeyCodec;

impl KeyCodec for V1KeyCodec {
    const VERSION: KeyVersion = KeyVersion::V1;
    type Parts = V1KeyParts;
    type Error = V1KeyError;

    // V1 payload layout:
    // [59:56] execution
    // [55:48] op
    // [47:16] selector
    // [15:00] reserved
    fn encode(parts: Self::Parts) -> KernelKey {
        let execution_bits = u64::from(encode_execution(parts.execution) & 0x0F);
        let op_bits = u64::from(parts.op.code());
        let selector_bits = u64::from(parts.selector);
        let payload = (execution_bits << 56) | (op_bits << 48) | (selector_bits << 16);
        compose_key(Self::VERSION, payload)
    }

    fn decode(key: KernelKey) -> Result<Self::Parts, Self::Error> {
        ensure_version(key, Self::VERSION)?;

        let payload = key_payload(key);
        let execution_code = ((payload >> 56) & 0x0F) as u8;
        let op_code = ((payload >> 48) & 0xFF) as u8;
        let selector = ((payload >> 16) & 0xFFFF_FFFF) as u32;

        let execution =
            decode_execution(execution_code).ok_or(V1KeyError::InvalidExecution(execution_code))?;
        let op = OpTag::from_code(op_code).ok_or(V1KeyError::InvalidOp(op_code))?;

        Ok(V1KeyParts {
            execution,
            op,
            selector,
        })
    }
}

impl V1KeyCodec {
    pub fn encode(parts: V1KeyParts) -> KernelKey {
        <Self as KeyCodec>::encode(parts)
    }

    pub fn decode(key: KernelKey) -> Result<V1KeyParts, V1KeyError> {
        <Self as KeyCodec>::decode(key)
    }
}

fn ensure_version(key: KernelKey, expected: KeyVersion) -> Result<(), V1KeyError> {
    let actual = key_version(key).map_err(V1KeyError::Version)?;
    if actual != expected {
        return Err(V1KeyError::VersionMismatch { expected, actual });
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
