use dense_impl::ops::fill::DenseFillError;
use op_contracts::FillOp;
use tensor::Tensor;

/// Op struct for fill operation. Carries the value to fill with.
pub struct Fill {
    pub value: f32,
}

#[derive(Debug)]
pub enum FillError {
    Dense(DenseFillError),
}

/// Fill all elements of `tensor` in-place with `value`.
pub fn fill(tensor: &mut Tensor, value: f32) -> Result<(), FillError> {
    match tensor {
        Tensor::Dense(dense) => dense.fill_inplace(value).map_err(FillError::Dense),
    }
}

/// Return a new tensor with all elements set to `value`, leaving the original unchanged.
pub fn fill_new(tensor: &Tensor, value: f32) -> Result<Tensor, FillError> {
    let mut t = tensor.clone();
    fill(&mut t, value)?;
    Ok(t)
}
