use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue};

use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};

pub fn input_key() -> ArgKey {
    ArgKey::new(ArgRole::Input, "in", ArgKind::Storage)
}

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn pos_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "pos", ArgKind::Usize)
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let in_key = input_key();
    let out_key = output_key();
    let pos_key = pos_key();

    let in_storage = args.args().require_as::<StorageValue>(&in_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at requires input storage arg '{}': {err:?}",
            in_key.tag().as_str()
        ))
    })?;

    let out_storage = args.args().require_as::<StorageValue>(&out_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at requires output storage arg '{}': {err:?}",
            out_key.tag().as_str()
        ))
    })?;

    let pos = *args.args().require_as::<usize>(&pos_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at requires usize param arg '{}': {err:?}",
            pos_key.tag().as_str()
        ))
    })?;

    if in_storage.dtype() != DType::F32 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.read_at requires input dtype F32, got {:?}",
            in_storage.dtype()
        )));
    }

    if out_storage.dtype() != DType::F32 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.read_at requires output dtype F32, got {:?}",
            out_storage.dtype()
        )));
    }

    let b = pos * std::mem::size_of::<f32>();
    let value_bytes = in_storage.buffer().with_read_bytes(|src| {
        [src[b], src[b + 1], src[b + 2], src[b + 3]]
    });

    out_storage.buffer().with_write_bytes(|dst| {
        dst[0..4].copy_from_slice(&value_bytes);
    });

    Ok(())
}
