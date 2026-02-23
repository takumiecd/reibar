use std::sync::{Mutex, OnceLock};

use execution::{
    CpuKernelArgs, KernelArgs, Storage, StorageAllocError, StorageContext, StorageRequest,
};
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{
    ArgKey, ArgKind, ArgRole, DType, KernelArg, Scalar, ScalarBuffer, ViewSpecError,
};

use crate::DenseTensorImpl;

/// Read a single element at the given multi-dimensional index.
pub fn exec_read(tensor: &DenseTensorImpl, indices: &[usize]) -> Result<Scalar, DenseAtError> {
    let dtype = tensor.dtype();
    let view_spec = tensor.view_spec().map_err(DenseAtError::InvalidViewSpec)?;
    let encoded_indices = encode_indices(indices)?;

    let context = StorageContext::from_execution_tag(tensor.execution_tag());
    let out = Storage::allocate(
        tensor.execution_tag(),
        &context,
        StorageRequest::new(dtype.size_bytes(), dtype),
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
        .insert(KernelArg::view_spec(view_spec_key(), view_spec))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::scalar_buffer(indices_key(), encoded_indices))
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
        .with_read_bytes(|bytes| scalar_from_element_bytes(dtype, bytes));

    Ok(value)
}

/// Write a single element at the given multi-dimensional index.
/// Uses interior mutability — `&DenseTensorImpl` is sufficient.
pub fn exec_write(
    tensor: &DenseTensorImpl,
    indices: &[usize],
    value: Scalar,
) -> Result<(), DenseAtError> {
    let dtype = tensor.dtype();
    let view_spec = tensor.view_spec().map_err(DenseAtError::InvalidViewSpec)?;
    let encoded_indices = encode_indices(indices)?;

    let out_cpu = match tensor.storage() {
        Storage::Cpu(storage) => storage.clone(),
    };

    let mut cpu_args = CpuKernelArgs::new();
    cpu_args
        .insert(KernelArg::storage(write_output_key(), out_cpu))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::view_spec(view_spec_key(), view_spec))
        .map_err(DenseAtError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::scalar_buffer(indices_key(), encoded_indices))
        .map_err(DenseAtError::KernelArgs)?;
    insert_write_value_arg(&mut cpu_args, dtype, value)?;
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
    ScalarDTypeMismatch { tensor_dtype: DType, value: Scalar },
    InvalidViewSpec(ViewSpecError),
    IndicesEncodingOverflow,
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

fn view_spec_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_spec", ArgKind::ViewSpec)
}

fn indices_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "indices", ArgKind::ScalarBuffer)
}

fn write_value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

fn scalar_from_element_bytes(dtype: DType, bytes: &[u8]) -> Scalar {
    dtype
        .decode_scalar(bytes)
        .expect("single-element output buffer must decode as requested dtype")
}

fn insert_write_value_arg(
    cpu_args: &mut CpuKernelArgs,
    dtype: DType,
    value: Scalar,
) -> Result<(), DenseAtError> {
    if value.dtype() != dtype {
        return Err(DenseAtError::ScalarDTypeMismatch {
            tensor_dtype: dtype,
            value,
        });
    }

    cpu_args
        .args_mut()
        .insert_scalar(write_value_key(dtype), value)
        .map_err(DenseAtError::KernelArgs)
}

fn encode_indices(indices: &[usize]) -> Result<ScalarBuffer, DenseAtError> {
    let mut bytes = Vec::with_capacity(
        indices
            .len()
            .checked_mul(std::mem::size_of::<i64>())
            .ok_or(DenseAtError::IndicesEncodingOverflow)?,
    );
    for &index in indices {
        let index = i64::try_from(index).map_err(|_| DenseAtError::IndicesEncodingOverflow)?;
        bytes.extend_from_slice(&index.to_ne_bytes());
    }
    ScalarBuffer::new(DType::I64, indices.len(), bytes).ok_or(DenseAtError::IndicesEncodingOverflow)
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
