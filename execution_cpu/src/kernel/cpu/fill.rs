use schema::{ArgKey, ArgRole};

use crate::{CpuKernelArgs, CpuKernelLaunchError};

pub fn register_output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out")
}

pub fn value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value")
}

pub fn launch(args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
    let out_key = register_output_key();
    let value_key = value_key();

    args.require_storage(&out_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.fill requires output storage arg '{}': {err:?}",
            out_key.tag().as_str()
        ))
    })?;

    let _value = args.require_f32(&value_key).map_err(|err| {
        CpuKernelLaunchError::new(format!(
            "cpu.fill requires f32 param arg '{}': {err:?}",
            value_key.tag().as_str()
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use schema::{ArgKey, ArgRole, KernelArg};

    use super::{launch, register_output_key, value_key};
    use crate::CpuKernelArgs;

    #[test]
    fn launch_accepts_expected_args() {
        let mut args = CpuKernelArgs::new();
        args.insert(KernelArg::storage(register_output_key(), ()))
            .expect("out insertion should succeed");
        args.insert(KernelArg::f32(value_key(), 3.0))
            .expect("value insertion should succeed");

        launch(&args).expect("fill launch should validate arguments");
    }

    #[test]
    fn launch_rejects_missing_value() {
        let mut args = CpuKernelArgs::new();
        args.insert(KernelArg::storage(register_output_key(), ()))
            .expect("out insertion should succeed");

        let err = launch(&args).expect_err("fill launch should fail without value");
        assert!(err.message().contains("value"));

        let missing = ArgKey::new(ArgRole::Param, "value");
        assert_eq!(missing, value_key());
    }
}
