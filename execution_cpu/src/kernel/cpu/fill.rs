use schema::{ArgKey, ArgKind, ArgRole, DType, StorageValue};

use crate::{CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
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
    let pattern = encoded_value(args, dtype)?;
    let element_bytes = dtype.size_bytes();
    out_storage
        .buffer()
        .with_write_bytes(|out| -> Result<(), CpuKernelLaunchError> {
            if out.len() % element_bytes != 0 {
                return Err(CpuKernelLaunchError::new(
                    format!(
                        "cpu.fill requires output byte-size to be multiple of element size {element_bytes}"
                    ),
                ));
            }

            for chunk in out.chunks_exact_mut(element_bytes) {
                chunk.copy_from_slice(&pattern);
            }
            Ok(())
        })?;

    Ok(())
}

fn encoded_value(args: &CpuKernelArgs, dtype: DType) -> Result<Vec<u8>, CpuKernelLaunchError> {
    let value_key = value_key(dtype);
    args.args()
        .require_scalar_bytes(&value_key, dtype)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill requires {:?} param arg '{}': {err:?}",
                dtype,
                value_key.tag().as_str()
            ))
        })
}

#[cfg(test)]
mod tests {
    use schema::{ArgKey, ArgKind, ArgRole, DType, KernelArg};

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
        args.insert(KernelArg::f32(value_key(DType::F32), 3.0))
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
        args.insert(KernelArg::i64(value_key(DType::I64), 7))
            .expect("value insertion should succeed");

        launch(&args, &CpuKernelLaunchConfig::new("cpu.fill"))
            .expect("fill launch should validate arguments");

        let values = out.buffer().with_read_bytes(decode_i64_vec);
        assert_eq!(values, vec![7, 7]);
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
