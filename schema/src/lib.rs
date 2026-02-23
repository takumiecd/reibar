mod arg;
mod args;
mod dtype;
mod encoded_scalar;
mod error;
mod key;
mod role;
mod scalar;
mod scalar_buffer;
mod stack_vector;
mod value;
mod view_spec;

pub use arg::KernelArg;
pub use args::KernelArgs;
pub use dtype::DType;
pub use encoded_scalar::EncodedScalar;
pub use error::KernelArgsError;
pub use key::{ArgKey, ArgTag};
pub use role::ArgRole;
pub use scalar::Scalar;
pub use scalar_buffer::{ScalarBuffer, ScalarBufferDecodeError, ScalarBufferElement};
pub use stack_vector::{StackVector, StackVectorError};
pub use value::{ArgKind, ArgValue, ArgValueAccess, StorageValue};
pub use view_spec::{ViewSpec, ViewSpecError};

#[cfg(test)]
mod tests {
    use super::{
        ArgKey, ArgKind, ArgRole, DType, KernelArg, KernelArgs, KernelArgsError, Scalar,
        ScalarBuffer, ScalarBufferDecodeError, StackVector, StorageValue, ViewSpec,
    };

    #[test]
    fn kernel_args_insert_and_require() {
        let mut args = KernelArgs::new();
        let input = ArgKey::new(ArgRole::Input, "dst", ArgKind::Storage);
        let alpha = ArgKey::new(ArgRole::Param, "alpha", ArgKind::Scalar(DType::F32));

        args.insert(KernelArg::storage(input.clone(), ()))
            .expect("storage insertion should succeed");
        args.insert(KernelArg::scalar(alpha.clone(), Scalar::F32(1.5)))
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
        let key = ArgKey::new(ArgRole::Param, "beta", ArgKind::Scalar(DType::I64));
        args.insert(KernelArg::scalar(key.clone(), Scalar::I64(42)))
            .expect("insertion should succeed");

        let result = args.require_as::<f32>(&key);
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: super::ArgKind::Scalar(DType::F32),
                actual: super::ArgKind::Scalar(DType::I64),
            })
        );
    }

    #[test]
    fn kernel_args_reject_key_and_value_type_mismatch_on_insert() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "alpha", ArgKind::Scalar(DType::F32));

        let result = args.insert(KernelArg::scalar(key.clone(), Scalar::I64(42)));
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: ArgKind::Scalar(DType::F32),
                actual: ArgKind::Scalar(DType::I64),
            })
        );
    }

    #[test]
    fn kernel_args_insert_and_require_scalar() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "value", ArgKind::Scalar(DType::I64));

        args.insert_scalar(key.clone(), Scalar::I64(42))
            .expect("scalar insertion should succeed");

        let value = args
            .require_scalar(&key, DType::I64)
            .expect("scalar retrieval should succeed");
        assert_eq!(value, Scalar::I64(42));
    }

    #[test]
    fn kernel_args_require_encoded_scalar() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "value", ArgKind::Scalar(DType::F32));

        args.insert_scalar(key.clone(), Scalar::F32(3.5))
            .expect("scalar insertion should succeed");

        let encoded = args
            .require_encoded_scalar(&key, DType::F32)
            .expect("scalar encoding retrieval should succeed");
        assert_eq!(encoded.as_slice(), 3.5f32.to_ne_bytes().as_slice());
    }

    #[test]
    fn kernel_args_insert_and_require_scalar_buffer() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "values", ArgKind::ScalarBuffer);
        let values = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64, 2, 3]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<u8>>(),
        )
        .expect("valid scalar buffer should be created");

        args.insert_scalar_buffer(key.clone(), values)
            .expect("scalar buffer insertion should succeed");

        let buf = args
            .require_scalar_buffer(&key)
            .expect("scalar buffer retrieval should succeed");
        assert_eq!(buf.dtype(), DType::I64);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.scalar_at(1), Some(Scalar::I64(2)));
    }

    #[test]
    fn kernel_args_reject_invalid_scalar_buffer_key_kind() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "values", ArgKind::Scalar(DType::I64));
        let values = ScalarBuffer::from_bytes(DType::I64, i64::to_ne_bytes(1).to_vec())
            .expect("valid scalar buffer should be created");

        let result = args.insert_scalar_buffer(key.clone(), values);
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: ArgKind::Scalar(DType::I64),
                actual: ArgKind::ScalarBuffer,
            })
        );
    }

    #[test]
    fn scalar_buffer_validates_len_and_decode() {
        let valid = ScalarBuffer::new(
            DType::F32,
            2,
            [1.0f32, 2.5f32]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect(),
        );
        assert!(valid.is_some());

        let invalid = ScalarBuffer::new(DType::F32, 2, vec![0u8; 7]);
        assert!(invalid.is_none());

        let not_aligned = ScalarBuffer::from_bytes(DType::I64, vec![0u8; 7]);
        assert!(not_aligned.is_none());
    }

    #[test]
    fn scalar_buffer_try_to_vec_supports_typed_decode() {
        let values = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64, 2, 3]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<u8>>(),
        )
        .expect("valid scalar buffer should be created");

        let decoded = values
            .try_to_vec::<i64>()
            .expect("typed decode should succeed");
        assert_eq!(decoded, vec![1, 2, 3]);
    }

    #[test]
    fn scalar_buffer_try_to_vec_rejects_dtype_mismatch() {
        let values = ScalarBuffer::from_bytes(
            DType::I64,
            [1i64].into_iter().flat_map(i64::to_ne_bytes).collect(),
        )
        .expect("valid scalar buffer should be created");

        let err = values
            .try_to_vec::<f32>()
            .expect_err("dtype mismatch should fail");
        assert_eq!(
            err,
            ScalarBufferDecodeError::DTypeMismatch {
                expected: DType::F32,
                actual: DType::I64
            }
        );
    }

    #[test]
    fn stack_vector_can_be_built_from_scalar_buffer() {
        let values = ScalarBuffer::from_bytes(
            DType::I64,
            [4i64, 5, 6]
                .into_iter()
                .flat_map(i64::to_ne_bytes)
                .collect::<Vec<u8>>(),
        )
        .expect("valid scalar buffer should be created");

        let stack = StackVector::<i64, 8>::try_from_scalar_buffer(&values)
            .expect("stack vector decode should succeed");
        assert_eq!(stack.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn kernel_args_insert_and_require_view_spec() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "view", ArgKind::ViewSpec);
        let view = ViewSpec::new(vec![2, 3], vec![3, 1], 4).expect("view should be valid");
        args.insert_view_spec(key.clone(), view.clone())
            .expect("view spec insertion should succeed");

        let got = args
            .require_view_spec(&key)
            .expect("view spec retrieval should succeed");
        assert_eq!(got, &view);
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
                .decode_scalar(encoded.as_slice())
                .expect("scalar decoding should succeed");
            assert_eq!(decoded, value);
        }
    }
}
