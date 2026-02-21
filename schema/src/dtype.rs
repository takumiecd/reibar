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
}
