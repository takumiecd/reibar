use std::sync::{Mutex, OnceLock};

use execution::{CpuKernelArgs, KernelArgs, Storage};
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg, Scalar, ViewSpec, ViewSpecError};

use crate::DenseTensorImpl;

pub fn exec(tensor: &mut DenseTensorImpl, value: Scalar) -> Result<(), DenseFillError> {
    dispatch_fill_kernel(tensor, value)
}

fn dispatch_fill_kernel(tensor: &DenseTensorImpl, value: Scalar) -> Result<(), DenseFillError> {
    let dtype = tensor.dtype();
    let mut cpu_args = CpuKernelArgs::new();
    let out = match tensor.storage() {
        Storage::Cpu(storage) => storage.clone(),
    };
    cpu_args
        .insert(KernelArg::storage(output_key(), out))
        .map_err(DenseFillError::KernelArgs)?;
    insert_fill_value_arg(&mut cpu_args, dtype, value)?;
    insert_fill_view_args(&mut cpu_args, tensor)?;
    let args = KernelArgs::Cpu(cpu_args);

    let seed = SeedSpec::V1(V1KeyParts {
        execution: tensor.execution_tag(),
        op: OpTag::Fill,
        selector: 0,
    });

    let dispatcher = dense_dispatcher();
    let mut dispatcher = dispatcher
        .lock()
        .map_err(|_| DenseFillError::DispatcherPoisoned)?;
    dispatcher
        .dispatch_seed(seed, &args)
        .map_err(DenseFillError::Dispatch)?;

    Ok(())
}

#[derive(Debug)]
pub enum DenseFillError {
    ScalarDTypeMismatch { tensor_dtype: DType, value: Scalar },
    InvalidViewSpec(ViewSpecError),
    DispatcherPoisoned,
    KernelArgs(schema::KernelArgsError),
    Dispatch(DispatchError),
}

fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

fn view_spec_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_spec", ArgKind::ViewSpec)
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

impl op_contracts::FillOp<DenseTensorImpl> for super::DenseOps {
    type Error = DenseFillError;
    fn fill_inplace(&self, tensor: &mut DenseTensorImpl, value: Scalar) -> Result<(), Self::Error> {
        exec(tensor, value)
    }
}

fn insert_fill_value_arg(
    cpu_args: &mut CpuKernelArgs,
    dtype: DType,
    value: Scalar,
) -> Result<(), DenseFillError> {
    if value.dtype() != dtype {
        return Err(DenseFillError::ScalarDTypeMismatch {
            tensor_dtype: dtype,
            value,
        });
    }

    cpu_args
        .args_mut()
        .insert_scalar(value_key(dtype), value)
        .map_err(DenseFillError::KernelArgs)
}

fn insert_fill_view_args(
    cpu_args: &mut CpuKernelArgs,
    tensor: &DenseTensorImpl,
) -> Result<(), DenseFillError> {
    cpu_args
        .insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(
                tensor.shape().to_vec(),
                tensor.strides().to_vec(),
                tensor.offset(),
            )
            .map_err(DenseFillError::InvalidViewSpec)?,
        ))
        .map_err(DenseFillError::KernelArgs)?;
    Ok(())
}
