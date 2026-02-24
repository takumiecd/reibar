use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};
use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue, ViewSpec};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn view_spec_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_spec", ArgKind::ViewSpec)
}

pub fn indices_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "indices", ArgKind::ScalarBuffer)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let out_key = output_key();
    let view_key = view_spec_key();
    let indices_key = indices_key();

    let out_storage = args
        .args()
        .require_as::<StorageValue>(&out_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.write_at requires output storage arg '{}': {err:?}",
                out_key.tag().as_str()
            ))
        })?;

    let view_spec = args
        .args()
        .require_as::<ViewSpec>(&view_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.write_at requires view-spec param arg '{}': {err:?}",
                view_key.tag().as_str()
            ))
        })?;
    let indices = args
        .args()
        .require_scalar_buffer(&indices_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.write_at requires scalar-buffer param arg '{}': {err:?}",
                indices_key.tag().as_str()
            ))
        })?;
    if indices.dtype() != DType::I64 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.write_at requires scalar-buffer arg '{}' to decode as i64 values: DTypeMismatch {{ expected: I64, actual: {:?} }}",
            indices_key.tag().as_str(),
            indices.dtype()
        )));
    }
    let expected_rank = view_spec.rank();
    let actual_rank = indices.len();
    if actual_rank != expected_rank {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.write_at invalid view/index metadata: IndicesRankMismatch {{ expected_rank: {expected_rank}, actual_rank: {actual_rank} }}"
        )));
    }
    let indices = indices.try_to_vec::<i64>().map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.write_at requires scalar-buffer arg '{}' to decode as i64 values: {err:?}",
            indices_key.tag().as_str()
        ))
    })?;
    let pos = view_spec.checked_pos_i64(&indices).map_err(|err| {
        CpuKernelLaunchError::new(format!("cpu.write_at invalid view/index metadata: {err:?}"))
    })?;

    let dtype = out_storage.dtype();
    let value_key = value_key(dtype);
    let encoded = args
        .args()
        .require_encoded_scalar(&value_key, dtype)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.write_at requires {:?} param arg '{}': {err:?}",
                dtype,
                value_key.tag().as_str()
            ))
        })?;
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
        slot.copy_from_slice(encoded.as_slice());
        Ok(())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use schema::{DType, KernelArg, Scalar, ScalarBuffer, ViewSpec};

    use super::{indices_key, launch, output_key, value_key, view_spec_key};
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![4], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view metadata insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[2]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(7.5)))
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![2], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view metadata insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[2]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(1.25)))
            .expect("value insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect_err("write_at launch should fail for out-of-bounds position");
        assert!(err.message().contains("IndexOutOfBounds"));
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
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![3], vec![1], 0).expect("view metadata should be valid"),
        ))
        .expect("view metadata insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[1]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::I64), Scalar::I64(9)))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect("write_at launch should succeed");

        let values = out.buffer().with_read_bytes(decode_i64_vec);
        assert_eq!(values, vec![0, 9, 0]);
    }

    #[test]
    fn launch_rejects_indices_rank_mismatch() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 4], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out))
            .expect("out insertion should succeed");
        args.insert(KernelArg::view_spec(
            view_spec_key(),
            ViewSpec::new(vec![1, 1], vec![1, 1], 0).expect("view metadata should be valid"),
        ))
        .expect("view metadata insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            indices_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[0]))
                .expect("indices metadata should be valid"),
        ))
        .expect("indices insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(1.25)))
            .expect("value insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.write_at"))
            .expect_err("write_at launch should fail for indices rank mismatch");
        assert!(err.message().contains("IndicesRankMismatch"));
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

    fn encode_i64_slice(values: &[i64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }
}
