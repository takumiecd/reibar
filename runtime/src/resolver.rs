use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use execution::KernelArgs;
use execution::{ExecutionTag, KernelArgs as ExecutionKernelArgs};
use schema::ArgRole;

use crate::{KernelKey, KeyVersion, OpTag};

pub trait KeyResolver {
    fn version(&self) -> KeyVersion;
    fn resolve(&self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V1KeyResolver;

impl KeyResolver for V1KeyResolver {
    fn version(&self) -> KeyVersion {
        KeyVersion::V1
    }

    fn resolve(&self, execution: ExecutionTag, op: OpTag, _args: &KernelArgs) -> KernelKey {
        KernelKey::v1(execution, op)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct V2KeyResolver;

impl KeyResolver for V2KeyResolver {
    fn version(&self) -> KeyVersion {
        KeyVersion::V2
    }

    fn resolve(&self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey {
        let signature_hash = hash_kernel_args(args);
        KernelKey::v2(execution, op, signature_hash)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolverKind {
    /// Production default: stable and minimal.
    #[default]
    V1,
    /// Opt-in path for future richer keying.
    ///
    /// Keep this explicit at call sites so migration is controlled per-op/per-flow.
    V2,
}

impl ResolverKind {
    pub fn version(self) -> KeyVersion {
        match self {
            Self::V1 => KeyVersion::V1,
            Self::V2 => KeyVersion::V2,
        }
    }

    pub fn resolve(self, execution: ExecutionTag, op: OpTag, args: &KernelArgs) -> KernelKey {
        match self {
            Self::V1 => V1KeyResolver.resolve(execution, op, args),
            Self::V2 => V2KeyResolver.resolve(execution, op, args),
        }
    }
}

fn hash_kernel_args(args: &ExecutionKernelArgs) -> u64 {
    let mut hasher = DefaultHasher::new();

    match args {
        ExecutionKernelArgs::Cpu(cpu_args) => {
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
        schema::ArgKind::F32 => 1,
        schema::ArgKind::I64 => 2,
        schema::ArgKind::Usize => 3,
        schema::ArgKind::Bool => 4,
    }
}
