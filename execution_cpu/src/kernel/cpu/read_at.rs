use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue, ViewSpec};

use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};

pub fn input_key() -> ArgKey {
    ArgKey::new(ArgRole::Input, "in", ArgKind::Storage)
}

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn view_spec_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_spec", ArgKind::ViewSpec)
}

pub fn indices_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "indices", ArgKind::ScalarBuffer)
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let in_key = input_key();
    let out_key = output_key();
    let view_key = view_spec_key();
    let indices_key = indices_key();

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

    let view_spec = args
        .args()
        .require_as::<ViewSpec>(&view_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.read_at requires view-spec param arg '{}': {err:?}",
                view_key.tag().as_str()
            ))
        })?;
    let indices_buffer = args
        .args()
        .require_scalar_buffer(&indices_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.read_at requires scalar-buffer param arg '{}': {err:?}",
                indices_key.tag().as_str()
            ))
        })?;
    if indices_buffer.dtype() != DType::I64 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.read_at requires scalar-buffer arg '{}' to decode as i64 values: DTypeMismatch {{ expected: I64, actual: {:?} }}",
            indices_key.tag().as_str(),
            indices_buffer.dtype()
        )));
    }
    let expected_rank = view_spec.rank();
    let actual_rank = indices_buffer.len();
    if actual_rank != expected_rank {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.read_at invalid view/index metadata: IndicesRankMismatch {{ expected_rank: {expected_rank}, actual_rank: {actual_rank} }}"
        )));
    }
    let indices = indices_buffer.try_to_vec::<i64>().map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.read_at requires scalar-buffer arg '{}' to decode as i64 values: {err:?}",
            indices_key.tag().as_str()
        ))
    })?;
    let pos = view_spec.checked_pos_i64(&indices).map_err(|err| {
        CpuKernelLaunchError::new(format!("cpu.read_at invalid view/index metadata: {err:?}"))
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
    use schema::{DType, KernelArg, ScalarBuffer, ViewSpec};

    use super::{indices_key, input_key, launch, output_key, view_spec_key};
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![3], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view spec insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[1]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");

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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![2], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view spec insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[2]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect_err("read_at launch should fail for out-of-bounds position");
        assert!(err.message().contains("IndexOutOfBounds"));
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![3], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view spec insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[1]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect("read_at launch should succeed");

        let value = out.buffer().with_read_bytes(|bytes| {
            i64::from_ne_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ])
        });
        assert_eq!(value, 22);
    }

    #[test]
    fn launch_rejects_indices_rank_mismatch() {
        let mut args = CpuKernelArgs::new();
        let input = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 4], DType::F32.alignment())
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![1, 1], vec![1, 1], 0).expect("view metadata should be valid"),
        ))
        .expect("view spec insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[0]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.read_at"))
            .expect_err("read_at launch should fail for indices rank mismatch");
        assert!(err.message().contains("IndicesRankMismatch"));
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
