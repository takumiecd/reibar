use schema::{ArgKey, ArgKind, ArgRole, StorageValue};

use crate::{CpuKernelArgs, CpuKernelLaunchError};

pub fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

pub fn value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", ArgKind::F32)
}

pub fn launch(args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
    let out_key = output_key();
    let value_key = value_key();

    let out_storage = args
        .args()
        .require_as::<StorageValue>(&out_key)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "cpu.fill requires output storage arg '{}': {err:?}",
                out_key.tag().as_str()
            ))
        })?;

    let value = *args.args().require_as::<f32>(&value_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.fill requires f32 param arg '{}': {err:?}",
            value_key.tag().as_str()
        ))
    })?;

    out_storage
        .with_write_bytes(|out| {
            if out.len() % std::mem::size_of::<f32>() != 0 {
                return Err(CpuKernelLaunchError::new(
                    "cpu.fill requires output byte-size to be multiple of f32 size",
                ));
            }

            let pattern = value.to_ne_bytes();
            for chunk in out.chunks_exact_mut(std::mem::size_of::<f32>()) {
                chunk.copy_from_slice(&pattern);
            }
            Ok(())
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use schema::{ArgKey, ArgKind, ArgRole, KernelArg};

    use super::{launch, output_key, value_key};
    use crate::{CpuKernelArgs, CpuStorage};

    #[test]
    fn launch_accepts_expected_args() {
        let mut args = CpuKernelArgs::new();
        let out = CpuStorage::new(vec![0u8; 16]);
        args.insert(KernelArg::storage(output_key(), out.clone()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::f32(value_key(), 3.0))
            .expect("value insertion should succeed");

        launch(&args).expect("fill launch should validate arguments");
        let values = out.with_read_bytes(decode_f32_vec);
        assert_eq!(values, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn launch_rejects_missing_value() {
        let mut args = CpuKernelArgs::new();
        args.insert(KernelArg::storage(output_key(), CpuStorage::new(vec![0u8; 8])))
            .expect("out insertion should succeed");

        let err = launch(&args).expect_err("fill launch should fail without value");
        assert!(err.message().contains("value"));

        let missing = ArgKey::new(ArgRole::Param, "value", ArgKind::F32);
        assert_eq!(missing, value_key());
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
}
