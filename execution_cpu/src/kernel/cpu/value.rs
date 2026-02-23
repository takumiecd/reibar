use schema::{ArgKey, ArgRole, DType, EncodedScalar};

use crate::{CpuKernelArgs, CpuKernelLaunchError};

pub(super) fn value_key(dtype: DType) -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", dtype.value_arg_kind())
}

pub(super) fn require_encoded_value(
    args: &CpuKernelArgs,
    dtype: DType,
    op_name: &str,
) -> Result<EncodedScalar, CpuKernelLaunchError> {
    let key = value_key(dtype);
    args.args()
        .require_encoded_scalar(&key, dtype)
        .map_err(|err| {
            CpuKernelLaunchError::new(format!(
                "{op_name} requires {:?} param arg '{}': {err:?}",
                dtype,
                key.tag().as_str()
            ))
        })
}
