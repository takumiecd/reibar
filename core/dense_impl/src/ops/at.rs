use execution::Storage;

use crate::tensor::DenseTensorImpl;

/// Read a single element at the given multi-dimensional index.
pub fn exec_read(tensor: &DenseTensorImpl, indices: &[usize]) -> Result<f32, DenseAtError> {
    let pos = validated_pos(tensor, indices)?;
    let b = pos * 4;
    let value = match tensor.storage() {
        Storage::Cpu(storage) => storage.buffer().with_read_bytes(|bytes| {
            f32::from_ne_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]])
        }),
    };
    Ok(value)
}

/// Write a single element at the given multi-dimensional index.
/// Uses interior mutability — `&DenseTensorImpl` is sufficient.
pub fn exec_write(
    tensor: &DenseTensorImpl,
    indices: &[usize],
    value: f32,
) -> Result<(), DenseAtError> {
    let pos = validated_pos(tensor, indices)?;
    let b = pos * 4;
    let encoded = value.to_ne_bytes();
    match tensor.storage() {
        Storage::Cpu(storage) => {
            storage.buffer().with_write_bytes(|slice| {
                slice[b..b + 4].copy_from_slice(&encoded);
            });
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum DenseAtError {
    RankMismatch { expected: usize, actual: usize },
    IndexOutOfBounds { dim: usize, index: usize, size: usize },
}

fn validated_pos(tensor: &DenseTensorImpl, indices: &[usize]) -> Result<usize, DenseAtError> {
    let shape = tensor.shape();
    if indices.len() != shape.len() {
        return Err(DenseAtError::RankMismatch {
            expected: shape.len(),
            actual: indices.len(),
        });
    }
    for (dim, (&idx, &size)) in indices.iter().zip(shape.iter()).enumerate() {
        if idx >= size {
            return Err(DenseAtError::IndexOutOfBounds { dim, index: idx, size });
        }
    }
    let pos = tensor.offset()
        + indices
            .iter()
            .zip(tensor.strides().iter())
            .map(|(i, s)| i * s)
            .sum::<usize>();
    Ok(pos)
}
