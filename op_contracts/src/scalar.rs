#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    F32(f32),
    I64(i64),
    U8(u8),
    Bool(bool),
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<i64> for Scalar {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<u8> for Scalar {
    fn from(value: u8) -> Self {
        Self::U8(value)
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
