use execution::Storage;

use crate::tensor::{flat_to_pos, DenseTensorImpl};

/// Return a CPU copy of `tensor`. For CPU tensors this clones the handle (shared storage).
/// Future device backends would transfer memory to a new CPU buffer here.
pub fn exec(tensor: &DenseTensorImpl) -> Result<DenseTensorImpl, DenseToHostError> {
    Ok(tensor.clone())
}

/// Read all elements of `tensor` in row-major order as `Vec<f32>`.
/// Used internally by `impl Debug for Tensor`.
pub fn exec_f32(tensor: &DenseTensorImpl) -> Result<Vec<f32>, DenseToHostError> {
    let numel = tensor.numel();
    let mut result = Vec::with_capacity(numel);
    match tensor.storage() {
        Storage::Cpu(storage) => {
            storage.buffer().with_read_bytes(|bytes| {
                for i in 0..numel {
                    let pos = flat_to_pos(i, tensor.shape(), tensor.strides(), tensor.offset());
                    let b = pos * 4;
                    result.push(f32::from_ne_bytes([
                        bytes[b],
                        bytes[b + 1],
                        bytes[b + 2],
                        bytes[b + 3],
                    ]));
                }
            });
        }
    }
    Ok(result)
}

#[derive(Debug)]
pub enum DenseToHostError {}
