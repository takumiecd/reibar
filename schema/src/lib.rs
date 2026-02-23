mod arg;
mod args;
mod dtype;
mod error;
mod key;
mod role;
mod scalar;
mod value;

pub use arg::KernelArg;
pub use args::KernelArgs;
pub use dtype::DType;
pub use error::KernelArgsError;
pub use key::{ArgKey, ArgTag};
pub use role::ArgRole;
pub use scalar::Scalar;
pub use value::{ArgKind, ArgValue, ArgValueAccess, StorageValue};

#[cfg(test)]
mod tests {
    use super::{
        ArgKey, ArgKind, ArgRole, DType, KernelArg, KernelArgs, KernelArgsError, Scalar,
        StorageValue,
    };

    #[test]
    fn kernel_args_insert_and_require() {
        let mut args = KernelArgs::new();
        let input = ArgKey::new(ArgRole::Input, "dst", ArgKind::Storage);
        let alpha = ArgKey::new(ArgRole::Param, "alpha", ArgKind::F32);

        args.insert(KernelArg::storage(input.clone(), ()))
            .expect("storage insertion should succeed");
        args.insert(KernelArg::f32(alpha.clone(), 1.5))
            .expect("param insertion should succeed");

        assert_eq!(args.len(), 2);
        assert!(args.require_as::<StorageValue>(&input).is_ok());
        assert_eq!(
            *args
                .require_as::<f32>(&alpha)
                .expect("f32 param should exist and have expected type"),
            1.5
        );
        assert!(args.require_as::<StorageValue>(&input).is_ok());
    }

    #[test]
    fn kernel_args_reject_duplicate_keys() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Input, "x", ArgKind::Storage);
        args.insert(KernelArg::storage(key.clone(), ()))
            .expect("first insertion should succeed");

        let result = args.insert(KernelArg::storage(key.clone(), ()));
        assert_eq!(result, Err(KernelArgsError::DuplicateKey { key }));
    }

    #[test]
    fn kernel_args_detect_type_mismatch() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "beta", ArgKind::I64);
        args.insert(KernelArg::i64(key.clone(), 42))
            .expect("insertion should succeed");

        let result = args.require_as::<f32>(&key);
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: super::ArgKind::F32,
                actual: super::ArgKind::I64,
            })
        );
    }

    #[test]
    fn kernel_args_reject_key_and_value_type_mismatch_on_insert() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "alpha", ArgKind::F32);

        let result = args.insert(KernelArg::i64(key.clone(), 42));
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: ArgKind::F32,
                actual: ArgKind::I64,
            })
        );
    }

    #[test]
    fn kernel_args_insert_and_require_scalar() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "value", ArgKind::I64);

        args.insert_scalar(key.clone(), Scalar::I64(42))
            .expect("scalar insertion should succeed");

        let value = args
            .require_scalar(&key, DType::I64)
            .expect("scalar retrieval should succeed");
        assert_eq!(value, Scalar::I64(42));
    }

    #[test]
    fn kernel_args_require_scalar_bytes() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "value", ArgKind::F32);

        args.insert_scalar(key.clone(), Scalar::F32(3.5))
            .expect("scalar insertion should succeed");

        let bytes = args
            .require_scalar_bytes(&key, DType::F32)
            .expect("scalar bytes retrieval should succeed");
        assert_eq!(bytes, 3.5f32.to_ne_bytes().to_vec());
    }

    #[test]
    fn dtype_encode_decode_scalar_roundtrip() {
        let values = [
            (DType::F32, Scalar::F32(1.25)),
            (DType::I64, Scalar::I64(-3)),
            (DType::U8, Scalar::U8(7)),
            (DType::Bool, Scalar::Bool(true)),
        ];

        for (dtype, value) in values {
            let encoded = dtype
                .encode_scalar(&value)
                .expect("scalar encoding should succeed");
            let decoded = dtype
                .decode_scalar(&encoded)
                .expect("scalar decoding should succeed");
            assert_eq!(decoded, value);
        }
    }
}
