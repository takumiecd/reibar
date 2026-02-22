use crate::builder::TensorBuilder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorTag {
    Dense,
}

impl TensorTag {
    pub fn builder(self, shape: Vec<usize>) -> TensorBuilder {
        match self {
            Self::Dense => TensorBuilder::dense(shape),
        }
    }
}
