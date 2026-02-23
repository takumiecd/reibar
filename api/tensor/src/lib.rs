mod builder;
mod tag;
mod tensor;

pub use builder::{TensorBuilder, TensorBuildError};
pub use tag::TensorTag;
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::{Tensor, TensorBuilder, TensorTag};

    #[test]
    fn tag_is_shared_between_builder_and_tensor() {
        let builder = TensorTag::Dense.builder(vec![2, 2]);
        assert_eq!(builder.tag(), TensorTag::Dense);

        let tensor = builder.build().expect("tensor build should succeed");
        assert_eq!(tensor.tag(), TensorTag::Dense);
    }

    #[test]
    fn dense_builder_builds_tensor() {
        let tensor = TensorBuilder::dense(vec![1, 3])
            .build()
            .expect("tensor build should succeed");

        match tensor {
            Tensor::Dense(inner) => {
                assert_eq!(inner.shape(), &[1, 3]);
            }
        }
    }
}
