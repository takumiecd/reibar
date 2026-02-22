mod dispatcher;
mod kernels;
mod key;
mod op;
mod registry;
mod version;

pub use dispatcher::{DispatchApi, DispatchRequest, SeedSpec};
pub use dispatcher::{DispatchError, Dispatcher};
pub use key::{
    KEY_PAYLOAD_MASK, KEY_VERSION_BITS, KEY_VERSION_MASK, KEY_VERSION_SHIFT, KernelKey, KeyError,
    KeyVersion, compose_key, key_payload, key_version, key_version_bits,
};
pub use op::OpTag;
pub use registry::{KernelRegistry, KernelRegistryConfig, KernelRegistryStats};
pub use version::{
    KeyCodec, KeyFallback, KeyResolver, ResolverKind, V1Fallback, V1KeyCodec, V1KeyError,
    V1KeyParts, V1KeyResolver, V2Fallback, V2KeyCodec, V2KeyError, V2KeyParts, V2KeyResolver,
    VersionError, next_fallback_key,
};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use execution::{
        CpuKernelArgs, CpuKernelLaunchConfig, CpuKernelLaunchError, CpuKernelMetadata,
        ExecutionTag, KernelArgs, KernelMetadata,
    };

    use super::{
        DispatchApi, DispatchError, DispatchRequest, Dispatcher, KernelRegistryConfig, KeyVersion,
        OpTag, ResolverKind, SeedSpec, V1KeyCodec, V1KeyParts, V2KeyCodec, V2KeyParts, key_payload,
        key_version,
    };

    static KERNEL_CALL_COUNT_V1_CONNECT: AtomicUsize = AtomicUsize::new(0);
    static KERNEL_CALL_COUNT_V1_FALLBACK: AtomicUsize = AtomicUsize::new(0);
    static KERNEL_CALL_COUNT_V1_SEED: AtomicUsize = AtomicUsize::new(0);
    static KERNEL_CALL_COUNT_V2_ACCEPT: AtomicUsize = AtomicUsize::new(0);
    static KERNEL_CALL_COUNT_V2_MISMATCH: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel_v1_connect(
        _args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V1_CONNECT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn test_fill_kernel_v1_fallback(
        _args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V1_FALLBACK.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn test_fill_kernel_v1_seed(
        _args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V1_SEED.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn test_fill_kernel_v2_accept(
        _args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V2_ACCEPT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn test_fill_kernel_v2_mismatch(
        _args: &CpuKernelArgs,
        _launch_config: &CpuKernelLaunchConfig,
    ) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V2_MISMATCH.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn dispatcher_connects_to_registry_and_launches() {
        KERNEL_CALL_COUNT_V1_CONNECT.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(
            KernelRegistryConfig {
                cache_capacity: 1,
                main_memory_capacity: 1,
            },
            KeyVersion::V1,
        );

        let fill_key = V1KeyCodec::encode(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 0,
        });
        let copy_key = V1KeyCodec::encode(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Copy,
            selector: 0,
        });

        dispatcher
            .register(
                fill_key,
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "fill_f32",
                    test_fill_kernel_v1_connect,
                )),
            )
            .expect("fill registration should succeed");
        dispatcher
            .register(
                copy_key,
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "copy_f32",
                    test_fill_kernel_v1_connect,
                )),
            )
            .expect("copy registration should succeed");

        let args = KernelArgs::Cpu(CpuKernelArgs::new());

        dispatcher
            .dispatch(DispatchRequest {
                key: fill_key,
                args: &args,
            })
            .expect("fill dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V1_CONNECT.load(Ordering::SeqCst), 1);
        assert_eq!(dispatcher.stats().secondary_hits, 1);

        dispatcher
            .dispatch(DispatchRequest {
                key: fill_key,
                args: &args,
            })
            .expect("fill dispatch should hit cache");
        assert_eq!(KERNEL_CALL_COUNT_V1_CONNECT.load(Ordering::SeqCst), 2);
        assert_eq!(dispatcher.stats().cache_hits, 1);

        dispatcher
            .dispatch(DispatchRequest {
                key: copy_key,
                args: &args,
            })
            .expect("copy dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V1_CONNECT.load(Ordering::SeqCst), 3);
        assert_eq!(dispatcher.stats().secondary_hits, 2);

        dispatcher
            .dispatch(DispatchRequest {
                key: fill_key,
                args: &args,
            })
            .expect("fill dispatch should rebuild from secondary");
        assert_eq!(KERNEL_CALL_COUNT_V1_CONNECT.load(Ordering::SeqCst), 4);
        assert_eq!(dispatcher.stats().secondary_hits, 3);
    }

    #[test]
    fn dispatcher_rejects_key_version_mismatch() {
        KERNEL_CALL_COUNT_V2_MISMATCH.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V1);
        let args = KernelArgs::Cpu(CpuKernelArgs::new());
        let key = V2KeyCodec::encode(V2KeyParts {
            op: OpTag::Fill,
            layout: 0,
            execution: ExecutionTag::Cpu,
            fingerprint: 1,
        });

        dispatcher
            .register(
                key,
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "fill_v2",
                    test_fill_kernel_v2_mismatch,
                )),
            )
            .expect_err("v2 key on v1 dispatcher should fail");

        let result = dispatcher.dispatch(DispatchRequest { key, args: &args });
        assert_eq!(
            result,
            Err(DispatchError::KeyVersionMismatch {
                dispatcher: KeyVersion::V1,
                key: KeyVersion::V2,
            })
        );
    }

    #[test]
    fn dispatcher_accepts_v2_when_configured() {
        KERNEL_CALL_COUNT_V2_ACCEPT.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V2);
        let args = KernelArgs::Cpu(CpuKernelArgs::new());
        let key = ResolverKind::V2.resolve(ExecutionTag::Cpu, OpTag::Fill, &args);

        dispatcher
            .register(
                key,
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "fill_v2",
                    test_fill_kernel_v2_accept,
                )),
            )
            .expect("v2 registration should succeed");

        dispatcher
            .dispatch(DispatchRequest { key, args: &args })
            .expect("v2 dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V2_ACCEPT.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn kernel_key_v1_and_v2_roundtrip() {
        let v1_key = V1KeyCodec::encode(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 7,
        });
        let v2_key = V2KeyCodec::encode(V2KeyParts {
            op: OpTag::Copy,
            layout: 3,
            execution: ExecutionTag::Cpu,
            fingerprint: 0xDEAD_BEEF,
        });

        assert_eq!(
            key_version(v1_key).expect("v1 key should decode version"),
            KeyVersion::V1
        );
        assert_eq!(
            key_version(v2_key).expect("v2 key should decode version"),
            KeyVersion::V2
        );

        let decoded_v1 = V1KeyCodec::decode(v1_key).expect("v1 decode should succeed");
        let decoded_v2 = V2KeyCodec::decode(v2_key).expect("v2 decode should succeed");

        assert_eq!(decoded_v1.execution, ExecutionTag::Cpu);
        assert_eq!(decoded_v1.op, OpTag::Fill);
        assert_eq!(decoded_v1.selector, 7);
        assert_eq!(decoded_v2.execution, ExecutionTag::Cpu);
        assert_eq!(decoded_v2.op, OpTag::Copy);
        assert_eq!(decoded_v2.layout, 3);
        assert_eq!(decoded_v2.fingerprint, 0xDEAD_BEEF);
        assert_ne!(key_payload(v1_key), key_payload(v2_key));
    }

    #[test]
    fn dispatcher_uses_incremental_v1_fallback() {
        KERNEL_CALL_COUNT_V1_FALLBACK.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V1)
            .with_max_fallback_steps(8);
        let args = KernelArgs::Cpu(CpuKernelArgs::new());
        let generic_key = V1KeyCodec::encode(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 0,
        });
        let optimized_seed = V1KeyCodec::encode(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 2,
        });

        dispatcher
            .register(
                generic_key,
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "fill_generic",
                    test_fill_kernel_v1_fallback,
                )),
            )
            .expect("generic v1 registration should succeed");

        dispatcher
            .dispatch(DispatchRequest {
                key: optimized_seed,
                args: &args,
            })
            .expect("dispatcher should fallback from selector=2 down to selector=0");

        assert_eq!(KERNEL_CALL_COUNT_V1_FALLBACK.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dispatch_seed_encodes_and_runs() {
        KERNEL_CALL_COUNT_V1_SEED.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V1);
        let args = KernelArgs::Cpu(CpuKernelArgs::new());
        let generic_seed = SeedSpec::V1(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 0,
        });
        let optimized_seed = SeedSpec::V1(V1KeyParts {
            execution: ExecutionTag::Cpu,
            op: OpTag::Fill,
            selector: 3,
        });

        dispatcher
            .register(
                generic_seed.encode(),
                KernelMetadata::Cpu(CpuKernelMetadata::new(
                    "fill_generic",
                    test_fill_kernel_v1_seed,
                )),
            )
            .expect("generic registration should succeed");

        dispatcher
            .dispatch_seed(optimized_seed, &args)
            .expect("dispatch_seed should encode and fallback to generic kernel");

        assert_eq!(KERNEL_CALL_COUNT_V1_SEED.load(Ordering::SeqCst), 1);
    }
}
