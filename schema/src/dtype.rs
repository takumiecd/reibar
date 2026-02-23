use crate::{ArgKind, Scalar};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::I64 => std::mem::size_of::<i64>(),
            Self::U8 => std::mem::size_of::<u8>(),
            Self::Bool => std::mem::size_of::<bool>(),
        }
    }

    pub fn alignment(self) -> usize {
        match self {
            Self::F32 => std::mem::align_of::<f32>(),
            Self::I64 => std::mem::align_of::<i64>(),
            Self::U8 => std::mem::align_of::<u8>(),
            Self::Bool => std::mem::align_of::<bool>(),
        }
    }

    pub fn value_arg_kind(self) -> ArgKind {
        ArgKind::Scalar(self)
    }

    pub fn encode_scalar(self, value: &Scalar) -> Option<Vec<u8>> {
        match (self, value) {
            (Self::F32, Scalar::F32(v)) => Some(v.to_ne_bytes().to_vec()),
            (Self::I64, Scalar::I64(v)) => Some(v.to_ne_bytes().to_vec()),
            (Self::U8, Scalar::U8(v)) => Some(vec![*v]),
            (Self::Bool, Scalar::Bool(v)) => Some(vec![u8::from(*v)]),
            _ => None,
        }
    }

    pub fn decode_scalar(self, bytes: &[u8]) -> Option<Scalar> {
        if bytes.len() != self.size_bytes() {
            return None;
        }

        let value = match self {
            Self::F32 => Scalar::F32(f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            Self::I64 => Scalar::I64(i64::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])),
            Self::U8 => Scalar::U8(bytes[0]),
            Self::Bool => Scalar::Bool(bytes[0] != 0),
        };

        Some(value)
    }
}
