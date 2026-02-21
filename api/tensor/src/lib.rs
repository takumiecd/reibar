use dense_impl::{DenseBuildError, DenseBuilder, DenseTensorImpl};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorTag {
    Dense,
}

#[derive(Debug, Clone)]
pub enum Tensor {
    Dense(DenseTensorImpl),
}

impl Tensor {
    pub fn tag(&self) -> TensorTag {
        match self {
            Self::Dense(_) => TensorTag::Dense,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(t) => t.shape(),
        }
    }
}

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

impl TensorTag {
    pub fn builder(self, shape: Vec<usize>) -> TensorBuilder {
        match self {
            Self::Dense => TensorBuilder::dense(shape),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorBuildError {
    Dense(DenseBuildError),
}

#[cfg(test)]
mod tests {
    use super::{Tensor, TensorBuilder, TensorTag};

    #[test]
    fn tag_is_shared_between_builder_and_tensor() {
        let builder = TensorTag::Dense.builder(vec![2, 2]);
        assert_eq!(builder.tag(), TensorTag::Dense);

        let tensor = builder.build().expect("tensor build should succeed");
        assert_eq!(tensor.tag(), TensorTag::Dense);
    }

    #[test]
    fn dense_builder_builds_tensor() {
        let tensor = TensorBuilder::dense(vec![1, 3])
            .build()
            .expect("tensor build should succeed");

        match tensor {
            Tensor::Dense(inner) => {
                assert_eq!(inner.shape(), &[1, 3]);
                assert_eq!(inner.data(), vec![0.0, 0.0, 0.0]);
            }
        }
    }
}
