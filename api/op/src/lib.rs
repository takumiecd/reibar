pub mod fill;

pub use fill::{fill, fill_new, Fill, FillError};

#[cfg(test)]
mod tests {
    use tensor::TensorBuilder;

    use super::{fill, fill_new};

    fn dense_tensor(shape: Vec<usize>) -> tensor::Tensor {
        TensorBuilder::dense(shape)
            .build()
            .expect("tensor build should succeed")
    }

    #[test]
    fn fill_succeeds_on_dense_tensor() {
        let mut tensor = dense_tensor(vec![2, 3]);
        fill(&mut tensor, 1.0).expect("fill should succeed");
    }

    #[test]
    fn fill_new_returns_tensor_with_same_shape() {
        let tensor = dense_tensor(vec![2, 3]);
        let filled = fill_new(&tensor, 1.0).expect("fill_new should succeed");
        assert_eq!(filled.shape(), &[2, 3]);
    }

    #[test]
    fn fill_new_does_not_alter_original_shape() {
        let tensor = dense_tensor(vec![4, 5]);
        let _ = fill_new(&tensor, 9.0).expect("fill_new should succeed");
        assert_eq!(tensor.shape(), &[4, 5]);
    }

    #[test]
    fn fill_applied_twice_succeeds() {
        let mut tensor = dense_tensor(vec![3]);
        fill(&mut tensor, 1.0).expect("first fill should succeed");
        fill(&mut tensor, 2.0).expect("second fill should succeed");
    }
}
