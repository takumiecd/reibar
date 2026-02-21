use std::sync::{Mutex, OnceLock};

use execution::{CpuKernelArgs, CpuStorage, ExecutionTag, KernelArgs, Storage};
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{ArgKey, ArgKind, ArgRole, KernelArg};

#[derive(Debug, Clone)]
pub struct DenseTensorImpl {
    shape: Vec<usize>,
    storage: Storage,
}

impl DenseTensorImpl {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    pub fn execution_tag(&self) -> ExecutionTag {
        self.storage.tag()
    }

    pub fn fill(&mut self, value: f32) -> Result<(), DenseOpError> {
        let mut cpu_args = CpuKernelArgs::new();
        let out = match &self.storage {
            Storage::Cpu(storage) => storage.clone(),
        };
        cpu_args
            .insert(KernelArg::storage(fill_output_key(), out))
            .map_err(DenseOpError::KernelArgs)?;
        cpu_args
            .insert(KernelArg::f32(fill_value_key(), value))
            .map_err(DenseOpError::KernelArgs)?;
        let args = KernelArgs::cpu(cpu_args);

        let seed = SeedSpec::V1(V1KeyParts {
            execution: self.execution_tag(),
            op: OpTag::Fill,
            selector: 0,
        });

        let dispatcher = dense_dispatcher();
        let mut dispatcher = dispatcher
            .lock()
            .map_err(|_| DenseOpError::DispatcherPoisoned)?;
        dispatcher
            .dispatch_seed(seed, &args)
            .map_err(DenseOpError::Dispatch)?;

        Ok(())
    }

    pub fn data(&self) -> Vec<f32> {
        match &self.storage {
            Storage::Cpu(storage) => storage.with_read(|slice| slice.to_vec()),
        }
    }
}

#[derive(Debug)]
pub enum DenseOpError {
    DispatcherPoisoned,
    KernelArgs(schema::KernelArgsError),
    Dispatch(DispatchError),
}

fn fill_output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn fill_value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", ArgKind::F32)
}

fn dense_dispatcher() -> &'static Mutex<Dispatcher> {
    static DISPATCHER: OnceLock<Mutex<Dispatcher>> = OnceLock::new();
    DISPATCHER.get_or_init(|| {
        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V1);
        dispatcher
            .initialize_builtins()
            .expect("dense builtins initialization must succeed");
        Mutex::new(dispatcher)
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBuilder {
    shape: Vec<usize>,
    data: Option<Vec<f32>>,
    fill_value: f32,
    execution_tag: ExecutionTag,
}

impl DenseBuilder {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            data: None,
            fill_value: 0.0,
            execution_tag: ExecutionTag::Cpu,
        }
    }

    pub fn with_data(mut self, data: Vec<f32>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn with_fill(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }

    pub fn with_execution(mut self, execution_tag: ExecutionTag) -> Self {
        self.execution_tag = execution_tag;
        self
    }

    pub fn build(self) -> Result<DenseTensorImpl, DenseBuildError> {
        let expected = element_count(&self.shape)?;
        let storage = match self.data {
            Some(data) => {
                if data.len() != expected {
                    return Err(DenseBuildError::DataLengthMismatch {
                        expected,
                        actual: data.len(),
                    });
                }
                match self.execution_tag {
                    ExecutionTag::Cpu => Storage::Cpu(CpuStorage::new(data)),
                }
            }
            None => match self.execution_tag {
                ExecutionTag::Cpu => Storage::Cpu(CpuStorage::new(vec![self.fill_value; expected])),
            },
        };

        Ok(DenseTensorImpl {
            shape: self.shape,
            storage,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DenseBuildError {
    DataLengthMismatch { expected: usize, actual: usize },
    ShapeOverflow,
}

fn element_count(shape: &[usize]) -> Result<usize, DenseBuildError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or(DenseBuildError::ShapeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use execution::ExecutionTag;

    use super::{DenseBuildError, DenseBuilder};

    #[test]
    fn build_from_fill() {
        let tensor = DenseBuilder::new(vec![2, 3])
            .with_fill(1.5)
            .build()
            .expect("dense build should succeed");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), vec![1.5, 1.5, 1.5, 1.5, 1.5, 1.5]);
        assert_eq!(tensor.execution_tag(), ExecutionTag::Cpu);
    }

    #[test]
    fn build_fails_on_data_len_mismatch() {
        let result = DenseBuilder::new(vec![2, 2])
            .with_data(vec![1.0, 2.0])
            .build();

        match result {
            Err(DenseBuildError::DataLengthMismatch { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 2);
            }
            other => panic!("unexpected build result: {other:?}"),
        }
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
            .with_data(vec![1.0, 2.0, 3.0, 4.0])
            .build()
            .expect("dense build should succeed");

        tensor.fill(9.5).expect("fill op should succeed");
        assert_eq!(tensor.data(), vec![9.5, 9.5, 9.5, 9.5]);
    }
}
