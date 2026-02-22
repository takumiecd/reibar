use dense_impl::ops::at::DenseAtError;
use dense_impl::DenseOps;
use execution::ExecutionTag;
use op_contracts::{ReadAtF32Op, WriteAtF32Op};
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

/// Read a single f32 element at the given indices. CPU tensors only.
pub fn at(tensor: &Tensor, indices: &[usize]) -> Result<f32, AtError> {
    match tensor {
        Tensor::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => {
            DenseOps.read_at_f32(dense, indices).map_err(AtError::Dense)
        }
        _ => Err(AtError::NotOnHost),
    }
}

/// Write a single f32 element at the given indices. CPU tensors only.
pub fn set_at(tensor: &mut Tensor, indices: &[usize], value: f32) -> Result<(), SetAtError> {
    match tensor {
        Tensor::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => {
            DenseOps.write_at_f32(dense, indices, value).map_err(SetAtError::Dense)
        }
        _ => Err(SetAtError::NotOnHost),
    }
}
