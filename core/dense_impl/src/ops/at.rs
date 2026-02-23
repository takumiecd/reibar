use std::sync::{Mutex, OnceLock};

use execution::{
    CpuKernelArgs, KernelArgs, Storage, StorageAllocError, StorageContext, StorageRequest,
};
use op_contracts::Scalar;
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg};

use crate::DenseTensorImpl;

/// Read a single element at the given multi-dimensional index.
pub fn exec_read(tensor: &DenseTensorImpl, indices: &[usize]) -> Result<Scalar, DenseAtError> {
    let pos = validated_pos(tensor, indices)?;

    let context = StorageContext::from_execution_tag(tensor.execution_tag());
    let out = Storage::allocate(
        tensor.execution_tag(),
        &context,
        StorageRequest::new(std::mem::size_of::<f32>(), DType::F32),
    )
    .map_err(DenseAtError::StorageAlloc)?;

    let (src_cpu, out_cpu) = match (tensor.storage(), &out) {
        (Storage::Cpu(src), Storage::Cpu(dst)) => (src.clone(), dst.clone()),
    };

    let mut cpu_args = CpuKernelArgs::new();
    cpu_args
        .insert(KernelArg::storage(read_input_key(), src_cpu))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::storage(read_output_key(), out_cpu.clone()))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::usize(pos_key(), pos))
        .map_err(DenseAtError::KernelArgs)?;
    let args = KernelArgs::Cpu(cpu_args);

    let seed = SeedSpec::V1(V1KeyParts {
        execution: tensor.execution_tag(),
        op: OpTag::ReadAt,
        selector: 0,
    });

    let dispatcher = dense_dispatcher();
    let mut dispatcher = dispatcher
        .lock()
        .map_err(|_| DenseAtError::DispatcherPoisoned)?;
    dispatcher
        .dispatch_seed(seed, &args)
        .map_err(DenseAtError::Dispatch)?;

    let value = out_cpu
        .buffer()
        .with_read_bytes(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));

    Ok(Scalar::F32(value))
}

/// Write a single element at the given multi-dimensional index.
/// Uses interior mutability — `&DenseTensorImpl` is sufficient.
pub fn exec_write(
    tensor: &DenseTensorImpl,
    indices: &[usize],
    value: Scalar,
) -> Result<(), DenseAtError> {
    let f32_value = match value {
        Scalar::F32(v) => v,
        other => return Err(DenseAtError::UnsupportedScalar(other)),
    };

    let pos = validated_pos(tensor, indices)?;

    let out_cpu = match tensor.storage() {
        Storage::Cpu(storage) => storage.clone(),
    };

    let mut cpu_args = CpuKernelArgs::new();
    cpu_args
        .insert(KernelArg::storage(write_output_key(), out_cpu))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::usize(pos_key(), pos))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::f32(write_value_key(), f32_value))
        .map_err(DenseAtError::KernelArgs)?;
    let args = KernelArgs::Cpu(cpu_args);

    let seed = SeedSpec::V1(V1KeyParts {
        execution: tensor.execution_tag(),
        op: OpTag::WriteAt,
        selector: 0,
    });

    let dispatcher = dense_dispatcher();
    let mut dispatcher = dispatcher
        .lock()
        .map_err(|_| DenseAtError::DispatcherPoisoned)?;
    dispatcher
        .dispatch_seed(seed, &args)
        .map_err(DenseAtError::Dispatch)?;

    Ok(())
}

#[derive(Debug)]
pub enum DenseAtError {
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        dim: usize,
        index: usize,
        size: usize,
    },
    UnsupportedScalar(Scalar),
    StorageAlloc(StorageAllocError),
    KernelArgs(schema::KernelArgsError),
    DispatcherPoisoned,
    Dispatch(DispatchError),
}

fn read_input_key() -> ArgKey {
    ArgKey::new(ArgRole::Input, "in", ArgKind::Storage)
}

fn read_output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn write_output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn pos_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "pos", ArgKind::Usize)
}

fn write_value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", ArgKind::F32)
}

fn validated_pos(tensor: &DenseTensorImpl, indices: &[usize]) -> Result<usize, DenseAtError> {
    let shape = tensor.shape();
    if indices.len() != shape.len() {
        return Err(DenseAtError::RankMismatch {
            expected: shape.len(),
            actual: indices.len(),
        });
    }
    for (dim, (&idx, &size)) in indices.iter().zip(shape.iter()).enumerate() {
        if idx >= size {
            return Err(DenseAtError::IndexOutOfBounds {
                dim,
                index: idx,
                size,
            });
        }
    }
    let pos = tensor.offset()
        + indices
            .iter()
            .zip(tensor.strides().iter())
            .map(|(i, s)| i * s)
            .sum::<usize>();
    Ok(pos)
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

impl op_contracts::ReadAtOp<DenseTensorImpl> for super::DenseOps {
    type Error = DenseAtError;
    fn read_at(&self, tensor: &DenseTensorImpl, indices: &[usize]) -> Result<Scalar, Self::Error> {
        exec_read(tensor, indices)
    }
}

impl op_contracts::WriteAtOp<DenseTensorImpl> for super::DenseOps {
    type Error = DenseAtError;
    fn write_at(
        &self,
        tensor: &DenseTensorImpl,
        indices: &[usize],
        value: Scalar,
    ) -> Result<(), Self::Error> {
        exec_write(tensor, indices, value)
    }
}
