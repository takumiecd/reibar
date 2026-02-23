use dense_impl::DenseOps;
use dense_impl::ops::at::DenseAtError;
use execution::ExecutionTag;
use op_contracts::{ReadAtOp, Scalar, WriteAtOp};
use tensor::Tensor;

#[derive(Debug)]
pub enum AtError {
    NotOnHost,
    Dense(DenseAtError),
}

#[derive(Debug)]
pub enum SetAtError {
    NotOnHost,
    Dense(DenseAtError),
}

/// Read a single element at the given indices. CPU tensors only.
pub fn at(tensor: &Tensor, indices: &[usize]) -> Result<Scalar, AtError> {
    match tensor {
        Tensor::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => {
            DenseOps.read_at(dense, indices).map_err(AtError::Dense)
        }
        _ => Err(AtError::NotOnHost),
    }
}

/// Write a single element at the given indices. CPU tensors only.
pub fn set_at(
    tensor: &mut Tensor,
    indices: &[usize],
    value: impl Into<Scalar>,
) -> Result<(), SetAtError> {
    let value = value.into();
    match tensor {
        Tensor::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => DenseOps
            .write_at(dense, indices, value)
            .map_err(SetAtError::Dense),
        _ => Err(SetAtError::NotOnHost),
    }
}
