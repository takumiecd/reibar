use execution::{ExecutionTag, Storage};

#[derive(Clone)]
pub struct DenseTensorImpl {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    storage: Storage,
}

impl DenseTensorImpl {
    pub(crate) fn new(shape: Vec<usize>, storage: Storage) -> Self {
        let strides = contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
            storage,
        }
    }

    pub(crate) fn with_view(
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
        storage: Storage,
    ) -> Self {
        Self {
            shape,
            strides,
            offset,
            storage,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn execution_tag(&self) -> ExecutionTag {
        self.storage.tag()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the tensor occupies a contiguous, offset-0 region of its storage.
    /// The fill kernel requires this.
    pub fn is_packed(&self) -> bool {
        self.offset == 0 && self.strides == contiguous_strides(&self.shape)
    }

    /// Read all elements in row-major order as `Vec<f32>`. Used by `impl Debug for Tensor`.
    pub fn read_all_f32(&self) -> Vec<f32> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        match &self.storage {
            Storage::Cpu(storage) => {
                storage.buffer().with_read_bytes(|bytes| {
                    for i in 0..numel {
                        let pos = flat_to_pos(i, &self.shape, &self.strides, self.offset);
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
        result
    }
}

/// Flat row-major index → storage element position accounting for strides and offset.
pub(crate) fn flat_to_pos(flat: usize, shape: &[usize], strides: &[usize], offset: usize) -> usize {
    let n = shape.len();
    let mut remaining = flat;
    let mut pos = offset;
    for i in (0..n).rev() {
        let idx = remaining % shape[i];
        remaining /= shape[i];
        pos += idx * strides[i];
    }
    pos
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
impl DenseTensorImpl {
    pub fn data(&self) -> Vec<f32> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        match &self.storage {
            Storage::Cpu(storage) => {
                storage.buffer().with_read_bytes(|bytes| {
                    for i in 0..numel {
                        let pos = flat_to_pos(i, &self.shape, &self.strides, self.offset);
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
        result
    }
}
