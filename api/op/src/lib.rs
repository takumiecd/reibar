pub mod at;
pub mod fill;
pub mod narrow;

pub use at::{AtError, SetAtError, at, set_at};
pub use fill::{Fill, FillError, fill, fill_new};
pub use narrow::{NarrowError, narrow};
pub use op_contracts::Scalar;

#[cfg(test)]
mod tests {
    use op_contracts::Scalar;
    use tensor::TensorBuilder;

    use super::{at, fill, fill_new, narrow, set_at};

    fn dense_tensor(shape: Vec<usize>) -> tensor::Tensor {
        TensorBuilder::dense(shape)
            .build()
            .expect("tensor build should succeed")
    }

    // --- fill ---

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
    fn fill_new_does_not_mutate_original() {
        let tensor = dense_tensor(vec![2, 3]);
        let _filled = fill_new(&tensor, 9.0).expect("fill_new should succeed");
        assert_eq!(at(&tensor, &[0, 0]).unwrap(), Scalar::F32(0.0));
        assert_eq!(at(&tensor, &[1, 2]).unwrap(), Scalar::F32(0.0));
    }

    #[test]
    fn fill_applied_twice_succeeds() {
        let mut tensor = dense_tensor(vec![3]);
        fill(&mut tensor, 1.0).expect("first fill should succeed");
        fill(&mut tensor, 2.0).expect("second fill should succeed");
    }

    // --- fill + at ---

    #[test]
    fn fill_then_readable_via_at() {
        let mut tensor = dense_tensor(vec![2, 3]);
        fill(&mut tensor, 1.5).expect("fill should succeed");
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(at(&tensor, &[i, j]).unwrap(), Scalar::F32(1.5));
            }
        }
    }

    #[test]
    fn fill_new_then_readable_via_at() {
        let tensor = dense_tensor(vec![3]);
        let filled = fill_new(&tensor, 7.0).expect("fill_new should succeed");
        for i in 0..3 {
            assert_eq!(at(&filled, &[i]).unwrap(), Scalar::F32(7.0));
        }
    }

    // --- at / set_at ---

    #[test]
    fn default_tensor_reads_zero_at_any_element() {
        let tensor = dense_tensor(vec![2, 3]);
        let v = at(&tensor, &[1, 2]).expect("at should succeed");
        assert_eq!(v, Scalar::F32(0.0));
    }

    #[test]
    fn set_at_then_at_returns_written_value() {
        let mut tensor = dense_tensor(vec![3, 3]);
        set_at(&mut tensor, &[1, 1], Scalar::F32(42.0)).expect("set_at should succeed");
        let v = at(&tensor, &[1, 1]).expect("at should succeed");
        assert_eq!(v, Scalar::F32(42.0));
    }

    #[test]
    fn set_at_does_not_affect_other_elements() {
        let mut tensor = dense_tensor(vec![2, 2]);
        set_at(&mut tensor, &[0, 1], Scalar::F32(5.0)).expect("set_at should succeed");
        assert_eq!(at(&tensor, &[0, 0]).unwrap(), Scalar::F32(0.0));
        assert_eq!(at(&tensor, &[1, 0]).unwrap(), Scalar::F32(0.0));
        assert_eq!(at(&tensor, &[1, 1]).unwrap(), Scalar::F32(0.0));
    }

    // --- narrow ---

    #[test]
    fn narrow_returns_view_with_reduced_shape() {
        let mut tensor = dense_tensor(vec![4, 3]);
        fill(&mut tensor, 1.0).expect("fill should succeed");
        let view = narrow(&tensor, 0, 1, 2).expect("narrow should succeed");
        assert_eq!(view.shape(), &[2, 3]);
    }

    #[test]
    fn narrow_view_reads_correct_values() {
        let mut tensor = dense_tensor(vec![4]);
        for i in 0..4 {
            set_at(&mut tensor, &[i], Scalar::F32(i as f32)).expect("set_at should succeed");
        }
        let view = narrow(&tensor, 0, 1, 2).expect("narrow should succeed");
        assert_eq!(at(&view, &[0]).unwrap(), Scalar::F32(1.0));
        assert_eq!(at(&view, &[1]).unwrap(), Scalar::F32(2.0));
    }

    #[test]
    fn narrow_out_of_bounds_returns_error() {
        let tensor = dense_tensor(vec![3]);
        assert!(narrow(&tensor, 0, 2, 2).is_err());
    }
}
