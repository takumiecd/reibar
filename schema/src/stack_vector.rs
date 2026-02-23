use crate::{ScalarBuffer, ScalarBufferDecodeError, ScalarBufferElement};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct StackVector<T, const MAX_LEN: usize> {
    len: u32,
    data: [T; MAX_LEN],
}

impl<T, const MAX_LEN: usize> StackVector<T, MAX_LEN>
where
    T: Copy + Default,
{
    pub fn len(&self) -> usize {
        usize::try_from(self.len).expect("u32 length must fit into usize")
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len()]
    }

    pub fn capacity(&self) -> usize {
        MAX_LEN
    }

    pub fn try_from_slice(values: &[T]) -> Result<Self, StackVectorError> {
        if values.len() > MAX_LEN {
            return Err(StackVectorError::TooLong {
                len: values.len(),
                max_len: MAX_LEN,
            });
        }
        let len = u32::try_from(values.len())
            .map_err(|_| StackVectorError::LengthOverflow { len: values.len() })?;

        let mut data = [T::default(); MAX_LEN];
        data[..values.len()].copy_from_slice(values);
        Ok(Self { len, data })
    }

    pub fn try_from_scalar_buffer(buffer: &ScalarBuffer) -> Result<Self, StackVectorError>
    where
        T: ScalarBufferElement,
    {
        let values = buffer
            .try_to_vec::<T>()
            .map_err(StackVectorError::ScalarBufferDecode)?;
        Self::try_from_slice(&values)
    }
}

impl<T, const MAX_LEN: usize> Default for StackVector<T, MAX_LEN>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            len: 0,
            data: [T::default(); MAX_LEN],
        }
    }
}

impl ScalarBuffer {
    pub fn try_to_stack_vector<T, const MAX_LEN: usize>(
        &self,
    ) -> Result<StackVector<T, MAX_LEN>, StackVectorError>
    where
        T: Copy + Default + ScalarBufferElement,
    {
        StackVector::try_from_scalar_buffer(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StackVectorError {
    TooLong { len: usize, max_len: usize },
    LengthOverflow { len: usize },
    ScalarBufferDecode(ScalarBufferDecodeError),
}

#[cfg(test)]
mod tests {
    use crate::{DType, ScalarBuffer};

    use super::{StackVector, StackVectorError};

    #[test]
    fn try_from_scalar_buffer_succeeds_for_matching_dtype() {
        let buffer = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64, 2, 3]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<u8>>(),
        )
        .expect("valid scalar buffer should be created");

        let values = StackVector::<i64, 4>::try_from_scalar_buffer(&buffer)
            .expect("i64 stack vector decoding should succeed");
        assert_eq!(values.len(), 3);
        assert_eq!(values.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn try_from_scalar_buffer_rejects_dtype_mismatch() {
        let buffer = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64].into_iter().flat_map(i64::to_ne_bytes).collect(),
        )
        .expect("valid scalar buffer should be created");

        let err = StackVector::<f32, 4>::try_from_scalar_buffer(&buffer)
            .expect_err("dtype mismatch should fail");
        assert!(matches!(
            err,
            StackVectorError::ScalarBufferDecode(crate::ScalarBufferDecodeError::DTypeMismatch {
                expected: DType::F32,
                actual: DType::I64
            })
        ));
    }

    #[test]
    fn try_from_scalar_buffer_rejects_too_long_input() {
        let buffer = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64, 2, 3]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<u8>>(),
        )
        .expect("valid scalar buffer should be created");

        let err = StackVector::<i64, 2>::try_from_scalar_buffer(&buffer)
            .expect_err("longer-than-capacity input should fail");
        assert_eq!(err, StackVectorError::TooLong { len: 3, max_len: 2 });
    }

    #[test]
    fn scalar_buffer_can_decode_to_stack_vector_directly() {
        let buffer = ScalarBuffer::from_bytes(
            DType::I64,
            [7i64, 8].into_iter().flat_map(i64::to_ne_bytes).collect(),
        )
        .expect("valid scalar buffer should be created");

        let values = buffer
            .try_to_stack_vector::<i64, 4>()
            .expect("direct stack vector decode should succeed");
        assert_eq!(values.as_slice(), &[7, 8]);
    }
}
