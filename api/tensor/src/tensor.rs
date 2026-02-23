use std::fmt;

use dense_impl::DenseTensorImpl;
use execution::ExecutionTag;
use schema::DType;

use crate::tag::TensorTag;

#[derive(Clone)]
pub enum Tensor {
    Dense(DenseTensorImpl),
}

impl Tensor {
    pub fn tag(&self) -> TensorTag {
        match self {
            Self::Dense(_) => TensorTag::Dense,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Dense(t) => t.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Dense(t) => t.dtype(),
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(dense) if dense.execution_tag() == ExecutionTag::Cpu => {
                let data = dense.read_all_scalars();
                f.debug_struct("Tensor")
                    .field("shape", &dense.shape())
                    .field("dtype", &dense.dtype())
                    .field("data", &data)
                    .finish()
            }
            Self::Dense(dense) => f
                .debug_struct("Tensor")
                .field("shape", &dense.shape())
                .field("dtype", &dense.dtype())
                .finish(),
        }
    }
}
