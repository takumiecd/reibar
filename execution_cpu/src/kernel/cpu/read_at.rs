use schema::{ArgKey, ArgKind, ArgRole, StorageValue};

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

    let in_storage = args
        .args()
        .require_as::<StorageValue>(&in_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.read_at requires input storage arg '{}': {err:?}",
                in_key.tag().as_str()
            ))
        })?;

    let out_storage = args
        .args()
        .require_as::<StorageValue>(&out_key)
        .map_err(|err| {
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

    if in_storage.dtype() != out_storage.dtype() {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.read_at requires matching input/output dtypes, got input={:?}, output={:?}",
            in_storage.dtype(),
            out_storage.dtype()
        )));
    }

    let element_bytes = in_storage.dtype().size_bytes();
    let b = pos.checked_mul(element_bytes).ok_or_else(|| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at position {pos} overflows byte offset calculation"
        ))
    })?;
    let e = b.checked_add(element_bytes).ok_or_else(|| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at position {pos} overflows byte range calculation"
        ))
    })?;

    let value_bytes = in_storage
        .buffer()
        .with_read_bytes(|src| {
            let Some(slot) = src.get(b..e) else {
                return Err(CpuKernelLaunchError::new(format!(
                    "cpu.read_at position {pos} (byte range {b}..{e}) is out of bounds for input buffer of {} bytes",
                    src.len()
                )));
            };
            Ok(slot.to_vec())
        })?;

    out_storage.buffer().with_write_bytes(|dst| {
        let Some(out_slot) = dst.get_mut(0..element_bytes) else {
            return Err(CpuKernelLaunchError::new(format!(
                "cpu.read_at requires output buffer of at least {element_bytes} bytes, got {} bytes",
                dst.len()
            )));
        };
        out_slot.copy_from_slice(&value_bytes);
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use schema::{DType, KernelArg};

    use super::{input_key, launch, output_key, pos_key};
    use crate::{CpuBuffer, CpuKernelArgs, CpuKernelLaunchConfig, CpuStorage};

    #[test]
    fn launch_reads_value_at_position() {
        let mut args = CpuKernelArgs::new();
        let input = CpuStorage::new(
            CpuBuffer::new_with_alignment(
                encode_f32_slice(&[1.0, 2.5, 3.0]),
                DType::F32.alignment(),
            )
            .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 4], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(input_key(), input))
            .expect("in insertion should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 1))
            .expect("pos insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect("read_at launch should succeed");

        let value = out
            .buffer()
            .with_read_bytes(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        assert_eq!(value, 2.5);
    }

    #[test]
    fn launch_rejects_out_of_bounds_position() {
        let mut args = CpuKernelArgs::new();
        let input = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 8], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 4], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(input_key(), input))
            .expect("in insertion should succeed");
        args.insert(KernelArg::storage(output_key(), out))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 2))
            .expect("pos insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect_err("read_at launch should fail for out-of-bounds position");
        assert!(err.message().contains("out of bounds"));
    }

    #[test]
    fn launch_supports_i64_dtype() {
        let mut args = CpuKernelArgs::new();
        let input = CpuStorage::new(
            CpuBuffer::new_with_alignment(encode_i64_slice(&[11, 22, 33]), DType::I64.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::I64,
        )
        .expect("typed storage creation should succeed");
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 8], DType::I64.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::I64,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(input_key(), input))
            .expect("in insertion should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 1))
            .expect("pos insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect("read_at launch should succeed");

        let value = out.buffer().with_read_bytes(|bytes| {
            i64::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
        assert_eq!(value, 22);
    }

    fn encode_f32_slice(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn encode_i64_slice(values: &[i64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}
