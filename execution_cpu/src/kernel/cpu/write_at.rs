use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue};

use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn pos_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "pos", ArgKind::Usize)
}

pub fn value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", ArgKind::F32)
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let out_key = output_key();
    let pos_key = pos_key();
    let value_key = value_key();

    let out_storage = args.args().require_as::<StorageValue>(&out_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at requires output storage arg '{}': {err:?}",
            out_key.tag().as_str()
        ))
    })?;

    let pos = *args.args().require_as::<usize>(&pos_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at requires usize param arg '{}': {err:?}",
            pos_key.tag().as_str()
        ))
    })?;

    let value = *args.args().require_as::<f32>(&value_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at requires f32 param arg '{}': {err:?}",
            value_key.tag().as_str()
        ))
    })?;

    if out_storage.dtype() != DType::F32 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.write_at requires output dtype F32, got {:?}",
            out_storage.dtype()
        )));
    }

    let b = pos * std::mem::size_of::<f32>();
    let encoded = value.to_ne_bytes();
    out_storage.buffer().with_write_bytes(|dst| {
        dst[b..b + 4].copy_from_slice(&encoded);
    });

    Ok(())
}
