use execution::ExecutionTag;

use crate::OpTag;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyVersion {
    /// Stable minimal key: execution + op.
    V1,
    /// Reserved for richer matching (dtype/layout/shape/context based signatures).
    ///
    /// Current implementation is intentionally conservative and may evolve.
    /// Compatibility is managed by this explicit version tag.
    V2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKeyV1 {
    execution: ExecutionTag,
    op: OpTag,
}

impl KernelKeyV1 {
    pub fn new(execution: ExecutionTag, op: OpTag) -> Self {
        Self { execution, op }
    }

    pub fn execution(&self) -> ExecutionTag {
        self.execution
    }

    pub fn op(&self) -> OpTag {
        self.op
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKeyV2 {
    execution: ExecutionTag,
    op: OpTag,
    signature_hash: u64,
}

impl KernelKeyV2 {
    pub fn new(execution: ExecutionTag, op: OpTag, signature_hash: u64) -> Self {
        Self {
            execution,
            op,
            signature_hash,
        }
    }

    pub fn execution(&self) -> ExecutionTag {
        self.execution
    }

    pub fn op(&self) -> OpTag {
        self.op
    }

    pub fn signature_hash(&self) -> u64 {
        self.signature_hash
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelKey {
    version: KeyVersion,
    payload: u128,
}

impl KernelKey {
    pub fn new(execution: ExecutionTag, op: OpTag) -> Self {
        Self::v1(execution, op)
    }

    pub fn v1(execution: ExecutionTag, op: OpTag) -> Self {
        let payload = encode_v1(KernelKeyV1::new(execution, op));
        Self {
            version: KeyVersion::V1,
            payload,
        }
    }

    pub fn v2(execution: ExecutionTag, op: OpTag, signature_hash: u64) -> Self {
        let payload = encode_v2(KernelKeyV2::new(execution, op, signature_hash));
        Self {
            version: KeyVersion::V2,
            payload,
        }
    }

    pub fn cpu(op: OpTag) -> Self {
        Self::v1(ExecutionTag::Cpu, op)
    }

    pub fn version(&self) -> KeyVersion {
        self.version
    }

    pub fn payload(&self) -> u128 {
        self.payload
    }

    pub fn execution(&self) -> ExecutionTag {
        match self.version {
            KeyVersion::V1 => decode_v1(self.payload).execution(),
            KeyVersion::V2 => decode_v2(self.payload).execution(),
        }
    }

    pub fn op(&self) -> OpTag {
        match self.version {
            KeyVersion::V1 => decode_v1(self.payload).op(),
            KeyVersion::V2 => decode_v2(self.payload).op(),
        }
    }

    pub fn as_v1(&self) -> Option<KernelKeyV1> {
        match self.version {
            KeyVersion::V1 => Some(decode_v1(self.payload)),
            KeyVersion::V2 => None,
        }
    }

    pub fn as_v2(&self) -> Option<KernelKeyV2> {
        match self.version {
            KeyVersion::V1 => None,
            KeyVersion::V2 => Some(decode_v2(self.payload)),
        }
    }
}

fn encode_v1(key: KernelKeyV1) -> u128 {
    let execution_bits = u128::from(encode_execution(key.execution()));
    let op_bits = u128::from(key.op().code());

    (execution_bits << 16) | op_bits
}

fn decode_v1(payload: u128) -> KernelKeyV1 {
    let op = OpTag::from_code((payload & 0xFFFF) as u16);
    let execution = decode_execution(((payload >> 16) & 0xFFFF) as u16);
    KernelKeyV1::new(execution, op)
}

fn encode_v2(key: KernelKeyV2) -> u128 {
    // Bit reservation for V2 payload:
    // [0..64)   signature_hash
    // [64..80)  op
    // [80..96)  execution
    // [96..128) reserved for future V2 expansion
    let signature_bits = u128::from(key.signature_hash());
    let op_bits = u128::from(key.op().code());
    let execution_bits = u128::from(encode_execution(key.execution()));

    signature_bits | (op_bits << 64) | (execution_bits << 80)
}

fn decode_v2(payload: u128) -> KernelKeyV2 {
    let signature_hash = (payload & u128::from(u64::MAX)) as u64;
    let op = OpTag::from_code(((payload >> 64) & 0xFFFF) as u16);
    let execution = decode_execution(((payload >> 80) & 0xFFFF) as u16);
    KernelKeyV2::new(execution, op, signature_hash)
}

fn encode_execution(tag: ExecutionTag) -> u16 {
    match tag {
        ExecutionTag::Cpu => 0,
    }
}

fn decode_execution(code: u16) -> ExecutionTag {
    match code {
        0 => ExecutionTag::Cpu,
        _ => unreachable!("invalid execution code generated internally: {code}"),
    }
}
