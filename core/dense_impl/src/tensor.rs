use execution::{ExecutionTag, Storage};
use op_contracts::FillOp;

#[derive(Debug, Clone)]
pub struct DenseTensorImpl {
    shape: Vec<usize>,
    storage: Storage,
}

impl DenseTensorImpl {
    pub(crate) fn new(shape: Vec<usize>, storage: Storage) -> Self {
        Self { shape, storage }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn execution_tag(&self) -> ExecutionTag {
        self.storage.tag()
    }
}

impl FillOp for DenseTensorImpl {
    type Error = crate::ops::fill::DenseFillError;

    fn fill_inplace(&mut self, value: f32) -> Result<(), Self::Error> {
        crate::ops::fill::exec(self, value)
    }
}

#[cfg(test)]
impl DenseTensorImpl {
    pub fn data(&self) -> Vec<f32> {
        use schema::DType;
        match &self.storage {
            Storage::Cpu(storage) => {
                debug_assert_eq!(storage.dtype(), DType::F32);
                storage.buffer().with_read_bytes(|bytes| {
                    let mut chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
                    let values: Vec<f32> = chunks
                        .by_ref()
                        .map(|chunk| {
                            f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        })
                        .collect();
                    debug_assert!(chunks.remainder().is_empty());
                    values
                })
            }
        }
    }
}
