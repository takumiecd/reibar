use dense_impl::DenseOps;
use dense_impl::ops::narrow::DenseNarrowError;
use op_contracts::NarrowOp;
use tensor::Tensor;

#[derive(Debug)]
pub enum NarrowError {
    Dense(DenseNarrowError),
}

/// Return a view of `tensor` along `dim`, starting at element `start` with length `len`.
/// Available for all backends; the returned tensor shares the underlying storage.
pub fn narrow(
    tensor: &Tensor,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<Tensor, NarrowError> {
    match tensor {
        Tensor::Dense(dense) => DenseOps
            .narrow(dense, dim, start, len)
            .map(Tensor::Dense)
            .map_err(NarrowError::Dense),
    }
}
