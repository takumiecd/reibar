use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue};

use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn pos_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "pos", ArgKind::Usize)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", value_arg_kind(dtype))
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let out_key = output_key();
    let pos_key = pos_key();

    let out_storage = args
        .args()
        .require_as::<StorageValue>(&out_key)
        .map_err(|err| {
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

    let dtype = out_storage.dtype();
    let encoded = encoded_value(args, dtype)?;
    let element_bytes = dtype.size_bytes();
    let b = pos.checked_mul(element_bytes).ok_or_else(|| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at position {pos} overflows byte offset calculation"
        ))
    })?;
    let e = b.checked_add(element_bytes).ok_or_else(|| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at position {pos} overflows byte range calculation"
        ))
    })?;

    out_storage.buffer().with_write_bytes(|dst| {
        let Some(slot) = dst.get_mut(b..e) else {
            return Err(CpuKernelLaunchError::new(format!(
                "cpu.write_at position {pos} (byte range {b}..{e}) is out of bounds for output buffer of {} bytes",
                dst.len()
            )));
        };
        slot.copy_from_slice(&encoded);
        Ok(())
    })?;

    Ok(())
}

fn value_arg_kind(dtype: DType) -> ArgKind {
    match dtype {
        DType::F32 => ArgKind::F32,
        DType::I64 => ArgKind::I64,
        DType::U8 => ArgKind::U8,
        DType::Bool => ArgKind::Bool,
    }
}

fn encoded_value(args: &CpuKernelArgs, dtype: DType) -> Result<Vec<u8>, CpuKernelLaunchError> {
    let value_key = value_key(dtype);
    match dtype {
        DType::F32 => args
            .args()
            .require_as::<f32>(&value_key)
            .map(|v| v.to_ne_bytes().to_vec())
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.write_at requires f32 param arg '{}': {err:?}",
                    value_key.tag().as_str()
                ))
            }),
        DType::I64 => args
            .args()
            .require_as::<i64>(&value_key)
            .map(|v| v.to_ne_bytes().to_vec())
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.write_at requires i64 param arg '{}': {err:?}",
                    value_key.tag().as_str()
                ))
            }),
        DType::U8 => args
            .args()
            .require_as::<u8>(&value_key)
            .map(|v| vec![*v])
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.write_at requires u8 param arg '{}': {err:?}",
                    value_key.tag().as_str()
                ))
            }),
        DType::Bool => args
            .args()
            .require_as::<bool>(&value_key)
            .map(|v| vec![u8::from(*v)])
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.write_at requires bool param arg '{}': {err:?}",
                    value_key.tag().as_str()
                ))
            }),
    }
}

#[cfg(test)]
mod tests {
    use schema::{DType, KernelArg};

    use super::{launch, output_key, pos_key, value_key};
    use crate::{CpuBuffer, CpuKernelArgs, CpuKernelLaunchConfig, CpuStorage};

    #[test]
    fn launch_writes_value_at_position() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 16], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 2))
            .expect("pos insertion should succeed");
        args.insert(KernelArg::f32(value_key(DType::F32), 7.5))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect("write_at launch should succeed");

        let values = out.buffer().with_read_bytes(decode_f32_vec);
        assert_eq!(values, vec![0.0, 0.0, 7.5, 0.0]);
    }

    #[test]
    fn launch_rejects_out_of_bounds_position() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 8], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 2))
            .expect("pos insertion should succeed");
        args.insert(KernelArg::f32(value_key(DType::F32), 1.25))
            .expect("value insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect_err("write_at launch should fail for out-of-bounds position");
        assert!(err.message().contains("out of bounds"));
    }

    #[test]
    fn launch_writes_i64_value() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 24], DType::I64.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::I64,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::usize(pos_key(), 1))
            .expect("pos insertion should succeed");
        args.insert(KernelArg::i64(value_key(DType::I64), 9))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect("write_at launch should succeed");

        let values = out.buffer().with_read_bytes(decode_i64_vec);
        assert_eq!(values, vec![0, 9, 0]);
    }

    fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
        let mut chunks = bytes.chunks_exact(std::mem::size_of::<f32>());
        let values: Vec<f32> = chunks
            .by_ref()
            .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert!(chunks.remainder().is_empty());
        values
    }

    fn decode_i64_vec(bytes: &[u8]) -> Vec<i64> {
        let mut chunks = bytes.chunks_exact(std::mem::size_of::<i64>());
        let values: Vec<i64> = chunks
            .by_ref()
            .map(|chunk| {
                i64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();
        assert!(chunks.remainder().is_empty());
        values
    }
}
