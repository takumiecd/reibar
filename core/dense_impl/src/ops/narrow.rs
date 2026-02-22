use crate::tensor::DenseTensorImpl;

/// Create a view of `tensor` along `dim`, starting at `start` with length `len`.
/// The returned tensor shares the same storage.
pub fn exec(
    tensor: &DenseTensorImpl,
    dim: usize,
    start: usize,
    len: usize,
) -> Result<DenseTensorImpl, DenseNarrowError> {
    let shape = tensor.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(DenseNarrowError::DimOutOfBounds { dim, ndim });
    }
    if start + len > shape[dim] {
        return Err(DenseNarrowError::RangeOutOfBounds {
            dim,
            start,
            len,
            size: shape[dim],
        });
    }

    let mut new_shape = shape.to_vec();
    new_shape[dim] = len;

    let new_strides = tensor.strides().to_vec();
    let new_offset = tensor.offset() + start * tensor.strides()[dim];

    Ok(DenseTensorImpl::with_view(
        new_shape,
        new_strides,
        new_offset,
        tensor.storage().clone(),
    ))
}

#[derive(Debug)]
pub enum DenseNarrowError {
    DimOutOfBounds { dim: usize, ndim: usize },
    RangeOutOfBounds { dim: usize, start: usize, len: usize, size: usize },
}
