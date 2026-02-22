use dense_impl::DenseTensorImpl;

use crate::tag::TensorTag;

#[derive(Debug, Clone)]
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
}
