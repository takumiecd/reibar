use std::sync::{Mutex, OnceLock};

use execution::{CpuKernelArgs, KernelArgs, Storage};
use op_contracts::Scalar;
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg};

use crate::{DenseTensorImpl, tensor::flat_to_pos};

pub fn exec(tensor: &mut DenseTensorImpl, value: Scalar) -> Result<(), DenseFillError> {
    if tensor.is_packed() {
        return dispatch_fill_kernel(tensor, value);
    }

    fill_view_cpu(tensor, value)
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

fn fill_view_cpu(tensor: &DenseTensorImpl, value: Scalar) -> Result<(), DenseFillError> {
    let dtype = tensor.dtype();
    let pattern = scalar_pattern(dtype, value)?;
    let element_bytes = dtype.size_bytes();
    let numel = tensor.numel();
    let shape = tensor.shape();
    let strides = tensor.strides();
    let offset = tensor.offset();

    let out = match tensor.storage() {
        Storage::Cpu(storage) => storage.clone(),
    };

    out.buffer().with_write_bytes(|dst| {
        for flat in 0..numel {
            let pos = flat_to_pos(flat, shape, strides, offset);
            let b = pos
                .checked_mul(element_bytes)
                .ok_or(DenseFillError::PositionOverflow { pos, element_bytes })?;
            let e = b
                .checked_add(element_bytes)
                .ok_or(DenseFillError::PositionOverflow { pos, element_bytes })?;
            let Some(slot) = dst.get_mut(b..e) else {
                return Err(DenseFillError::ViewOutOfBounds {
                    pos,
                    range_start: b,
                    range_end: e,
                    buffer_len: dst.len(),
                });
            };
            slot.copy_from_slice(&pattern);
        }
        Ok(())
    })
}

#[derive(Debug)]
pub enum DenseFillError {
    ScalarDTypeMismatch {
        tensor_dtype: DType,
        value: Scalar,
    },
    PositionOverflow {
        pos: usize,
        element_bytes: usize,
    },
    ViewOutOfBounds {
        pos: usize,
        range_start: usize,
        range_end: usize,
        buffer_len: usize,
    },
    DispatcherPoisoned,
    KernelArgs(schema::KernelArgsError),
    Dispatch(DispatchError),
}

fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", scalar_arg_kind(dtype))
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

fn scalar_arg_kind(dtype: DType) -> ArgKind {
    match dtype {
        DType::F32 => ArgKind::F32,
        DType::I64 => ArgKind::I64,
        DType::U8 => ArgKind::U8,
        DType::Bool => ArgKind::Bool,
    }
}

fn insert_fill_value_arg(
    cpu_args: &mut CpuKernelArgs,
    dtype: DType,
    value: Scalar,
) -> Result<(), DenseFillError> {
    let insertion = match (dtype, value) {
        (DType::F32, Scalar::F32(v)) => cpu_args.insert(KernelArg::f32(value_key(dtype), v)),
        (DType::I64, Scalar::I64(v)) => cpu_args.insert(KernelArg::i64(value_key(dtype), v)),
        (DType::U8, Scalar::U8(v)) => cpu_args.insert(KernelArg::u8(value_key(dtype), v)),
        (DType::Bool, Scalar::Bool(v)) => cpu_args.insert(KernelArg::bool(value_key(dtype), v)),
        (tensor_dtype, other) => {
            return Err(DenseFillError::ScalarDTypeMismatch {
                tensor_dtype,
                value: other,
            });
        }
    };

    insertion.map_err(DenseFillError::KernelArgs)
}

fn scalar_pattern(dtype: DType, value: Scalar) -> Result<Vec<u8>, DenseFillError> {
    match (dtype, value) {
        (DType::F32, Scalar::F32(v)) => Ok(v.to_ne_bytes().to_vec()),
        (DType::I64, Scalar::I64(v)) => Ok(v.to_ne_bytes().to_vec()),
        (DType::U8, Scalar::U8(v)) => Ok(vec![v]),
        (DType::Bool, Scalar::Bool(v)) => Ok(vec![u8::from(v)]),
        (tensor_dtype, other) => Err(DenseFillError::ScalarDTypeMismatch {
            tensor_dtype,
            value: other,
        }),
    }
}
