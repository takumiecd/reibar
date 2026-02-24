use crate::{DType, Scalar};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarBuffer {
    dtype: DType,
    len: usize,
    bytes: Vec<u8>,
}

impl ScalarBuffer {
    pub fn new(dtype: DType, len: usize, bytes: Vec<u8>) -> Option<Self> {
        let expected_len = len.checked_mul(dtype.size_bytes())?;
        if bytes.len() != expected_len {
            return None;
        }

        Some(Self { dtype, len, bytes })
    }

    pub fn from_bytes(dtype: DType, bytes: Vec<u8>) -> Option<Self> {
        let element_size = dtype.size_bytes();
        if !bytes.len().is_multiple_of(element_size) {
            return None;
        }
        let len = bytes.len() / element_size;
        Some(Self { dtype, len, bytes })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn scalar_at(&self, index: usize) -> Option<Scalar> {
        if index >= self.len {
            return None;
        }

        let element_size = self.dtype.size_bytes();
        let start = index.checked_mul(element_size)?;
        let end = start.checked_add(element_size)?;
        self.dtype.decode_scalar(&self.bytes[start..end])
    }

    pub fn try_to_vec<T>(&self) -> Result<Vec<T>, ScalarBufferDecodeError>
    where
        T: ScalarBufferElement,
    {
        if self.dtype != T::DTYPE {
            return Err(ScalarBufferDecodeError::DTypeMismatch {
                expected: T::DTYPE,
                actual: self.dtype,
            });
        }

        let mut values = Vec::with_capacity(self.len);
        for index in 0..self.len {
            let scalar = self
                .scalar_at(index)
                .ok_or(ScalarBufferDecodeError::DecodeFailed { index })?;
            let value =
                T::from_scalar(scalar).ok_or(ScalarBufferDecodeError::DecodeFailed { index })?;
            values.push(value);
        }
        Ok(values)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarBufferDecodeError {
    DTypeMismatch { expected: DType, actual: DType },
    DecodeFailed { index: usize },
}

pub trait ScalarBufferElement: Sized {
    const DTYPE: DType;

    fn from_scalar(value: Scalar) -> Option<Self>;
}

impl ScalarBufferElement for f32 {
    const DTYPE: DType = DType::F32;

    fn from_scalar(value: Scalar) -> Option<Self> {
        match value {
            Scalar::F32(v) => Some(v),
            _ => None,
        }
    }
}

impl ScalarBufferElement for i64 {
    const DTYPE: DType = DType::I64;

    fn from_scalar(value: Scalar) -> Option<Self> {
        match value {
            Scalar::I64(v) => Some(v),
            _ => None,
        }
    }
}

impl ScalarBufferElement for u8 {
    const DTYPE: DType = DType::U8;

    fn from_scalar(value: Scalar) -> Option<Self> {
        match value {
            Scalar::U8(v) => Some(v),
            _ => None,
        }
    }
}

impl ScalarBufferElement for bool {
    const DTYPE: DType = DType::Bool;

    fn from_scalar(value: Scalar) -> Option<Self> {
        match value {
            Scalar::Bool(v) => Some(v),
            _ => None,
        }
    }
}
