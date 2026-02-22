use dense_impl::{DenseBuilder, DenseBuildError};

use crate::tag::TensorTag;
use crate::tensor::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorBuilder {
    Dense(DenseBuilder),
}

impl TensorBuilder {
    pub fn dense(shape: Vec<usize>) -> Self {
        Self::Dense(DenseBuilder::new(shape))
    }

    pub fn tag(&self) -> TensorTag {
        match self {
            Self::Dense(_) => TensorTag::Dense,
        }
    }

    pub fn build(self) -> Result<Tensor, TensorBuildError> {
        match self {
            Self::Dense(builder) => builder
                .build()
                .map(Tensor::Dense)
                .map_err(TensorBuildError::Dense),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorBuildError {
    Dense(DenseBuildError),
}
