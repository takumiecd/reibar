use dense_impl::ops::to_host::DenseToHostError;
use dense_impl::DenseOps;
use execution::ExecutionTag;
use op_contracts::ToHostOp;
use tensor::Tensor;

#[derive(Debug)]
pub enum ToHostError {
    NotOnHost,
    Dense(DenseToHostError),
}

/// Return a CPU `Tensor` equivalent to `tensor`.
/// For CPU tensors the result shares the same underlying storage (read via `at`).
/// Returns `ToHostError::NotOnHost` for non-CPU tensors (future: device → host copy).
pub fn to_host(tensor: &Tensor) -> Result<Tensor, ToHostError> {
    match tensor {
        Tensor::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => {
            DenseOps.to_host(dense).map(Tensor::Dense).map_err(ToHostError::Dense)
        }
        _ => Err(ToHostError::NotOnHost),
    }
}
