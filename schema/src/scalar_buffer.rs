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
        if bytes.len() % element_size != 0 {
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
}
