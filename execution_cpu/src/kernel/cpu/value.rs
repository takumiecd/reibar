use schema::{ArgKey, ArgRole, DType};

use crate::{CpuKernelArgs, CpuKernelLaunchError};

pub(super) fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

pub(super) fn require_encoded_value(
    args: &CpuKernelArgs,
    dtype: DType,
    op_name: &str,
) -> Result<Vec<u8>, CpuKernelLaunchError> {
    let key = value_key(dtype);
    args.args()
        .require_scalar_bytes(&key, dtype)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "{op_name} requires {:?} param arg '{}': {err:?}",
                dtype,
                key.tag().as_str()
            ))
        })
}
