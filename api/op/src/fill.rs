use dense_impl::ops::fill::DenseFillError;
use dense_impl::{DenseBuildError, DenseBuilder, DenseOps};
use op_contracts::{FillOp, Scalar};
use tensor::Tensor;

/// Op struct for fill operation. Carries the value to fill with.
pub struct Fill {
    pub value: Scalar,
}

#[derive(Debug)]
pub enum FillError {
    Dense(DenseFillError),
    Build(DenseBuildError),
}

/// Fill all elements of `tensor` in-place with `value`.
pub fn fill(tensor: &mut Tensor, value: impl Into<Scalar>) -> Result<(), FillError> {
    let value = value.into();
    match tensor {
        Tensor::Dense(dense) => DenseOps
            .fill_inplace(dense, value)
            .map_err(FillError::Dense),
    }
}

/// Return a new tensor with all elements set to `value`, leaving the original unchanged.
/// Allocates fresh storage so the original tensor is not affected.
pub fn fill_new(tensor: &Tensor, value: impl Into<Scalar>) -> Result<Tensor, FillError> {
    let value = value.into();
    let mut t = match tensor {
        Tensor::Dense(dense) => DenseBuilder::new(dense.shape().to_vec())
            .with_execution(dense.execution_tag())
            .with_dtype(dense.dtype())
            .build()
            .map(Tensor::Dense)
            .map_err(FillError::Build)?,
    };
    fill(&mut t, value)?;
    Ok(t)
}
