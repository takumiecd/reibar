use execution::{ExecutionTag, Storage, StorageAllocError, StorageContext, StorageRequest};
use schema::DType;

use crate::tensor::DenseTensorImpl;

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBuilder {
    shape: Vec<usize>,
    execution_tag: ExecutionTag,
    dtype: DType,
}

impl DenseBuilder {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            execution_tag: ExecutionTag::Cpu,
            dtype: DType::F32,
        }
    }

    pub fn with_execution(mut self, execution_tag: ExecutionTag) -> Self {
        self.execution_tag = execution_tag;
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Result<DenseTensorImpl, DenseBuildError> {
        let expected = element_count(&self.shape)?;
        let bytes = expected
            .checked_mul(self.dtype.size_bytes())
            .ok_or(DenseBuildError::ShapeOverflow)?;
        let request = StorageRequest::new(bytes, self.dtype);
        let storage_context = StorageContext::from_execution_tag(self.execution_tag);
        let storage = Storage::allocate(self.execution_tag, &storage_context, request)
            .map_err(DenseBuildError::StorageAlloc)?;
        Ok(DenseTensorImpl::new(self.shape, storage))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DenseBuildError {
    StorageAlloc(StorageAllocError),
    ShapeOverflow,
}

fn element_count(shape: &[usize]) -> Result<usize, DenseBuildError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or(DenseBuildError::ShapeOverflow)
    })
}
