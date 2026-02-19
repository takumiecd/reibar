mod arg;
mod args;
mod error;
mod key;
mod role;
mod value;

pub use arg::KernelArg;
pub use args::KernelArgs;
pub use error::KernelArgsError;
pub use key::{ArgKey, ArgTag};
pub use role::ArgRole;
pub use value::{ArgKind, ArgValue, ArgValueAccess, StorageValue};

#[cfg(test)]
mod tests {
    use super::{ArgKey, ArgRole, KernelArg, KernelArgs, KernelArgsError, StorageValue};

    #[test]
    fn kernel_args_insert_and_require() {
        let mut args = KernelArgs::new();
        let input = ArgKey::new(ArgRole::Input, "dst");
        let alpha = ArgKey::new(ArgRole::Param, "alpha");

        args.insert(KernelArg::storage(input.clone(), ()))
            .expect("storage insertion should succeed");
        args.insert(KernelArg::f32(alpha.clone(), 1.5))
            .expect("param insertion should succeed");

        assert_eq!(args.len(), 2);
        assert!(args.require_storage(&input).is_ok());
        assert_eq!(
            *args
                .require_f32(&alpha)
                .expect("f32 param should exist and have expected type"),
            1.5
        );
        assert!(args.require_as::<StorageValue>(&input).is_ok());
    }

    #[test]
    fn kernel_args_reject_duplicate_keys() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Input, "x");
        args.insert(KernelArg::storage(key.clone(), ()))
            .expect("first insertion should succeed");

        let result = args.insert(KernelArg::storage(key.clone(), ()));
        assert_eq!(result, Err(KernelArgsError::DuplicateKey { key }));
    }

    #[test]
    fn kernel_args_detect_type_mismatch() {
        let mut args: KernelArgs<()> = KernelArgs::new();
        let key = ArgKey::new(ArgRole::Param, "beta");
        args.insert(KernelArg::i64(key.clone(), 42))
            .expect("insertion should succeed");

        let result = args.require_f32(&key);
        assert_eq!(
            result,
            Err(KernelArgsError::TypeMismatch {
                key,
                expected: super::ArgKind::F32,
                actual: super::ArgKind::I64,
            })
        );
    }
}
