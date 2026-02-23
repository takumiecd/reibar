use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use execution::{ExecutionTag, KernelArgs};
use schema::ArgRole;

use crate::version::api::KeyResolver;
use crate::version::v2::codec::{V2KeyCodec, V2KeyParts};
use crate::{KernelKey, KeyVersion, OpTag};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2KeyResolver;

impl KeyResolver for V2KeyResolver {
    fn version(&self) -> KeyVersion {
        KeyVersion::V2
    }

    fn resolve(&self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey {
        let fingerprint = hash_kernel_args(args);
        V2KeyCodec::encode(V2KeyParts {
            op,
            layout: 0,
            execution,
            fingerprint,
        })
    }
}

fn hash_kernel_args(args: &KernelArgs) -> u64 {
    let mut hasher = DefaultHasher::new();

    match args {
        KernelArgs::Cpu(cpu_args) => {
            cpu_args.context().worker_threads().hash(&mut hasher);

            let mut normalized: Vec<(u8, String, u8)> = cpu_args
                .args()
                .iter()
                .map(|arg| {
                    (
                        role_code(arg.key().role()),
                        arg.key().tag().as_str().to_string(),
                        kind_code(arg.value().kind()),
                    )
                })
                .collect();
            normalized.sort();
            normalized.hash(&mut hasher);
        }
    }

    hasher.finish()
}

fn role_code(role: ArgRole) -> u8 {
    match role {
        ArgRole::Input => 0,
        ArgRole::Output => 1,
        ArgRole::Temp => 2,
        ArgRole::Param => 3,
        ArgRole::Context => 4,
    }
}

fn kind_code(kind: schema::ArgKind) -> u8 {
    match kind {
        schema::ArgKind::Storage => 0,
        schema::ArgKind::Scalar(schema::DType::F32) => 1,
        schema::ArgKind::Scalar(schema::DType::I64) => 2,
        schema::ArgKind::Usize => 3,
        schema::ArgKind::Scalar(schema::DType::Bool) => 4,
        schema::ArgKind::Scalar(schema::DType::U8) => 5,
        schema::ArgKind::ScalarBuffer => 6,
    }
}

#[cfg(test)]
mod tests {
    use schema::ArgKind;

    use super::kind_code;

    #[test]
    fn kind_code_assignments_are_stable() {
        assert_eq!(kind_code(ArgKind::Storage), 0);
        assert_eq!(kind_code(ArgKind::Scalar(schema::DType::F32)), 1);
        assert_eq!(kind_code(ArgKind::Scalar(schema::DType::I64)), 2);
        assert_eq!(kind_code(ArgKind::Usize), 3);
        assert_eq!(kind_code(ArgKind::Scalar(schema::DType::Bool)), 4);
        assert_eq!(kind_code(ArgKind::Scalar(schema::DType::U8)), 5);
        assert_eq!(kind_code(ArgKind::ScalarBuffer), 6);
    }
}
