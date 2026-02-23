use execution::{ExecutionTag, Storage};
use schema::{DType, Scalar};

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

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when the tensor occupies a contiguous, offset-0 region of its storage.
    /// The fill kernel requires this.
    pub fn is_packed(&self) -> bool {
        self.offset == 0 && has_contiguous_strides(&self.shape, &self.strides)
    }

    pub fn read_all_scalars(&self) -> Vec<Scalar> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        let is_contiguous = has_contiguous_strides(&self.shape, &self.strides);
        let dtype = self.dtype();
        let element_bytes = dtype.size_bytes();

        match &self.storage {
            Storage::Cpu(storage) => {
                storage.buffer().with_read_bytes(|bytes| {
                    if is_contiguous {
                        for i in 0..numel {
                            let pos = self.offset + i;
                            result.push(read_scalar_at(bytes, dtype, element_bytes, pos));
                        }
                    } else {
                        for i in 0..numel {
                            let pos = flat_to_pos(i, &self.shape, &self.strides, self.offset);
                            result.push(read_scalar_at(bytes, dtype, element_bytes, pos));
                        }
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

fn has_contiguous_strides(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    let mut expected = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if stride != expected {
            return false;
        }
        expected = match expected.checked_mul(dim) {
            Some(next) => next,
            None => return false,
        };
    }
    true
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

fn read_scalar_at(bytes: &[u8], dtype: DType, element_bytes: usize, pos: usize) -> Scalar {
    let b = pos * element_bytes;
    let e = b + element_bytes;
    dtype
        .decode_scalar(&bytes[b..e])
        .expect("element byte slice must decode as tensor dtype")
}

#[cfg(test)]
impl DenseTensorImpl {
    pub fn data(&self) -> Vec<f32> {
        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        let is_contiguous = has_contiguous_strides(&self.shape, &self.strides);
        match &self.storage {
            Storage::Cpu(storage) => {
                storage.buffer().with_read_bytes(|bytes| {
                    if is_contiguous {
                        for i in 0..numel {
                            let b = (self.offset + i) * std::mem::size_of::<f32>();
                            result.push(f32::from_ne_bytes([
                                bytes[b],
                                bytes[b + 1],
                                bytes[b + 2],
                                bytes[b + 3],
                            ]));
                        }
                    } else {
                        for i in 0..numel {
                            let pos = flat_to_pos(i, &self.shape, &self.strides, self.offset);
                            let b = pos * std::mem::size_of::<f32>();
                            result.push(f32::from_ne_bytes([
                                bytes[b],
                                bytes[b + 1],
                                bytes[b + 2],
                                bytes[b + 3],
                            ]));
                        }
                    }
                });
            }
        }
        result
    }
}
