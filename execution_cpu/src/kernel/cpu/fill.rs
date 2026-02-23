use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};
use schema::{ArgKey, ArgKind, ArgRole, DType, ScalarBuffer, StorageValue};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

fn view_shape_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_shape", ArgKind::ScalarBuffer)
}

fn view_strides_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_strides", ArgKind::ScalarBuffer)
}

fn view_offset_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "view_offset", ArgKind::Usize)
}

pub fn launch(
    args: &CpuKernelArgs,
    _launch_config: &CpuKernelLaunchConfig,
) -> Result<(), CpuKernelLaunchError> {
    let out_key = output_key();

    let out_storage = args
        .args()
        .require_as::<StorageValue>(&out_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill requires output storage arg '{}': {err:?}",
                out_key.tag().as_str()
            ))
        })?;

    let dtype = out_storage.dtype();
    let value_key = value_key(dtype);
    let pattern = args
        .args()
        .require_encoded_scalar(&value_key, dtype)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill requires {:?} param arg '{}': {err:?}",
                dtype,
                value_key.tag().as_str()
            ))
        })?;
    let element_bytes = dtype.size_bytes();
    let view_spec = parse_view_spec(args)?;
    out_storage
        .buffer()
        .with_write_bytes(|out| -> Result<(), CpuKernelLaunchError> {
            match &view_spec {
                Some(spec) => {
                    for flat in 0..spec.numel {
                        let pos = flat_to_pos(flat, &spec.shape, &spec.strides, spec.offset)?;
                        let b = pos.checked_mul(element_bytes).ok_or_else(|| {
                            CpuKernelLaunchError::new(format!(
                                "cpu.fill view position {pos} overflows byte offset calculation"
                            ))
                        })?;
                        let e = b.checked_add(element_bytes).ok_or_else(|| {
                            CpuKernelLaunchError::new(format!(
                                "cpu.fill view position {pos} overflows byte range calculation"
                            ))
                        })?;
                        let Some(slot) = out.get_mut(b..e) else {
                            return Err(CpuKernelLaunchError::new(format!(
                                "cpu.fill view position {pos} (byte range {b}..{e}) is out of bounds for output buffer of {} bytes",
                                out.len()
                            )));
                        };
                        slot.copy_from_slice(pattern.as_slice());
                    }
                }
                None => {
                    if out.len() % element_bytes != 0 {
                        return Err(CpuKernelLaunchError::new(format!(
                            "cpu.fill requires output byte-size to be multiple of element size {element_bytes}"
                        )));
                    }

                    for chunk in out.chunks_exact_mut(element_bytes) {
                        chunk.copy_from_slice(pattern.as_slice());
                    }
                }
            }
            Ok(())
        })?;

    Ok(())
}

#[derive(Debug, Clone)]
struct FillViewSpec {
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    numel: usize,
}

fn parse_view_spec(args: &CpuKernelArgs) -> Result<Option<FillViewSpec>, CpuKernelLaunchError> {
    let shape_key = view_shape_key();
    let strides_key = view_strides_key();
    let offset_key = view_offset_key();

    let has_shape = args.args().iter().any(|arg| arg.key() == &shape_key);
    let has_strides = args.args().iter().any(|arg| arg.key() == &strides_key);
    let has_offset = args.args().iter().any(|arg| arg.key() == &offset_key);

    if !(has_shape || has_strides || has_offset) {
        return Ok(None);
    }

    if !(has_shape && has_strides && has_offset) {
        return Err(CpuKernelLaunchError::new(
            "cpu.fill view mode requires all view metadata args: view_shape, view_strides, view_offset",
        ));
    }

    let shape = decode_usize_buffer(
        args.args()
            .require_scalar_buffer(&shape_key)
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.fill requires scalar-buffer param arg '{}': {err:?}",
                    shape_key.tag().as_str()
                ))
            })?,
        shape_key.tag().as_str(),
    )?;
    let strides = decode_usize_buffer(
        args.args()
            .require_scalar_buffer(&strides_key)
            .map_err(|err| {
                CpuKernelLaunchError::new(format!(
                    "cpu.fill requires scalar-buffer param arg '{}': {err:?}",
                    strides_key.tag().as_str()
                ))
            })?,
        strides_key.tag().as_str(),
    )?;
    let offset = *args
        .args()
        .require_as::<usize>(&offset_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill requires usize param arg '{}': {err:?}",
                offset_key.tag().as_str()
            ))
        })?;

    if shape.len() != strides.len() {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.fill view metadata requires same rank for shape/strides, got shape={} and strides={}",
            shape.len(),
            strides.len()
        )));
    }

    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim).ok_or(()));
    let numel = numel.map_err(|_| {
        CpuKernelLaunchError::new("cpu.fill view shape overflows numel calculation")
    })?;

    Ok(Some(FillViewSpec {
        shape,
        strides,
        offset,
        numel,
    }))
}

fn decode_usize_buffer(
    buffer: &ScalarBuffer,
    arg_tag: &str,
) -> Result<Vec<usize>, CpuKernelLaunchError> {
    if buffer.dtype() != DType::I64 {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.fill requires scalar-buffer arg '{}' to use i64 dtype, got {:?}",
            arg_tag,
            buffer.dtype()
        )));
    }

    let mut chunks = buffer.bytes().chunks_exact(std::mem::size_of::<i64>());
    let mut values = Vec::with_capacity(buffer.len());
    for chunk in chunks.by_ref() {
        let value = i64::from_ne_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        if value < 0 {
            return Err(CpuKernelLaunchError::new(format!(
                "cpu.fill requires scalar-buffer arg '{}' to contain non-negative i64 values",
                arg_tag
            )));
        }
        values.push(usize::try_from(value).map_err(|_| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill scalar-buffer arg '{}' value {value} overflows usize",
                arg_tag
            ))
        })?);
    }
    if !chunks.remainder().is_empty() {
        return Err(CpuKernelLaunchError::new(format!(
            "cpu.fill scalar-buffer arg '{}' has invalid byte length",
            arg_tag
        )));
    }

    Ok(values)
}

fn flat_to_pos(
    flat: usize,
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> Result<usize, CpuKernelLaunchError> {
    let mut remaining = flat;
    let mut pos = offset;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim == 0 {
            return Err(CpuKernelLaunchError::new(
                "cpu.fill encountered zero-sized dim while materializing view indices",
            ));
        }
        let idx = remaining % dim;
        remaining /= dim;
        let step = idx.checked_mul(strides[i]).ok_or_else(|| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill view index calculation overflows at dim {i}"
            ))
        })?;
        pos = pos.checked_add(step).ok_or_else(|| {
            CpuKernelLaunchError::new("cpu.fill view position calculation overflows")
        })?;
    }
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg, Scalar, ScalarBuffer};

    use super::{launch, output_key, value_key};
    use crate::{CpuBuffer, CpuKernelArgs, CpuKernelLaunchConfig, CpuStorage};

    #[test]
    fn launch_accepts_expected_args() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 16], DType::F32.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(3.0)))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect("fill launch should validate arguments");
        let values = out.buffer().with_read_bytes(decode_f32_vec);
        assert_eq!(values, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn launch_rejects_missing_value() {
        let mut args = CpuKernelArgs::new();
        args.insert(KernelArg::storage(
            output_key(),
            CpuStorage::new(
                CpuBuffer::new_with_alignment(vec![0u8; 8], DType::F32.alignment())
                    .expect("aligned buffer creation should succeed"),
                DType::F32,
            )
            .expect("typed storage creation should succeed"),
        ))
        .expect("out insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect_err("fill launch should fail without value");
        assert!(err.message().contains("value"));

        let missing = ArgKey::new(ArgRole::Param, "value", ArgKind::Scalar(DType::F32));
        assert_eq!(missing, value_key(DType::F32));
    }

    #[test]
    fn launch_supports_i64_dtype() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(vec![0u8; 16], DType::I64.alignment())
                .expect("aligned buffer creation should succeed"),
            DType::I64,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::I64), Scalar::I64(7)))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect("fill launch should validate arguments");

        let values = out.buffer().with_read_bytes(decode_i64_vec);
        assert_eq!(values, vec![7, 7]);
    }

    #[test]
    fn launch_fills_only_view_region_when_view_metadata_is_provided() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(
            CpuBuffer::new_with_alignment(
                encode_f32_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                DType::F32.alignment(),
            )
            .expect("aligned buffer creation should succeed"),
            DType::F32,
        )
        .expect("typed storage creation should succeed");
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(9.0)))
            .expect("value insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            super::view_shape_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[2, 2]))
                .expect("shape metadata buffer should be valid"),
        ))
        .expect("shape metadata insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            super::view_strides_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[3, 1]))
                .expect("strides metadata buffer should be valid"),
        ))
        .expect("strides metadata insertion should succeed");
        args.insert(KernelArg::usize(super::view_offset_key(), 1))
            .expect("offset metadata insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect("fill launch with view metadata should succeed");

        let values = out.buffer().with_read_bytes(decode_f32_vec);
        assert_eq!(values, vec![0.0, 9.0, 9.0, 3.0, 9.0, 9.0]);
    }

    #[test]
    fn launch_rejects_partial_view_metadata() {
        let mut args = CpuKernelArgs::new();
        args.insert(KernelArg::storage(
            output_key(),
            CpuStorage::new(
                CpuBuffer::new_with_alignment(vec![0u8; 16], DType::F32.alignment())
                    .expect("aligned buffer creation should succeed"),
                DType::F32,
            )
            .expect("typed storage creation should succeed"),
        ))
        .expect("out insertion should succeed");
        args.insert(KernelArg::scalar(value_key(DType::F32), Scalar::F32(1.0)))
            .expect("value insertion should succeed");
        args.insert(KernelArg::scalar_buffer(
            super::view_shape_key(),
            ScalarBuffer::from_bytes(DType::I64, encode_i64_slice(&[2, 2]))
                .expect("shape metadata buffer should be valid"),
        ))
        .expect("shape metadata insertion should succeed");

        let err = launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect_err("fill launch should fail with partial view metadata");
        assert!(err.message().contains("requires all view metadata"));
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
