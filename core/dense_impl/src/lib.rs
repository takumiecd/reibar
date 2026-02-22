mod builder;
pub mod ops;
mod tensor;

pub use builder::{DenseBuilder, DenseBuildError};
pub use ops::DenseOps;
pub use tensor::DenseTensorImpl;

#[cfg(test)]
mod tests {
    use execution::ExecutionTag;
    use op_contracts::FillOp;

    use super::{ops::DenseOps, DenseBuilder, DenseBuildError};

    #[test]
    fn build_then_fill() {
        let mut tensor = DenseBuilder::new(vec![2, 3])
            .build()
            .expect("dense build should succeed");

        DenseOps.fill_inplace(&mut tensor, 1.5).expect("fill should succeed");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.strides(), &[3, 1]);
        assert_eq!(tensor.data(), vec![1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
        assert_eq!(tensor.execution_tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn build_sets_storage_with_execution_tag() {
        let tensor = DenseBuilder::new(vec![1, 2])
            .with_execution(ExecutionTag::Cpu)
            .build()
            .expect("dense build should succeed");

        assert_eq!(tensor.storage().tag(), ExecutionTag::Cpu);
        assert_eq!(tensor.data(), vec![0.0, 0.0]);
    }

    #[test]
    fn fill_op_dispatches_and_updates_storage() {
        let mut tensor = DenseBuilder::new(vec![2, 2])
            .build()
            .expect("dense build should succeed");

        DenseOps.fill_inplace(&mut tensor, 9.5).expect("fill op should succeed");
        assert_eq!(tensor.data(), vec![9.5, 9.5, 9.5, 9.5]);
    }

    #[test]
    fn shape_overflow_is_detected() {
        let result = DenseBuilder::new(vec![usize::MAX, 2]).build();
        match result {
            Err(DenseBuildError::ShapeOverflow) => {}
            Err(_) => panic!("expected ShapeOverflow, got a different error"),
            Ok(_) => panic!("expected ShapeOverflow, but build succeeded"),
        }
    }
}
