mod dispatcher;
mod key;
mod op;
mod registry;
mod resolver;

pub use dispatcher::{DispatchError, Dispatcher};
pub use key::{KernelKey, KernelKeyV1, KernelKeyV2, KeyVersion};
pub use op::OpTag;
pub use registry::{
    KernelRegistry, KernelRegistryConfig, KernelRegistryError, KernelRegistryStats,
};
pub use resolver::{KeyResolver, ResolverKind, V1KeyResolver, V2KeyResolver};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use execution::{
        CpuKernelArgs, CpuKernelLaunchError, ExecutionTag, KernelArgs, KernelMetadata,
    };

    use super::{Dispatcher, KernelKey, KernelRegistryConfig, KeyVersion, OpTag, ResolverKind};

    static KERNEL_CALL_COUNT_V1: AtomicUsize = AtomicUsize::new(0);
    static KERNEL_CALL_COUNT_V2: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel_v1(_args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V1.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn test_fill_kernel_v2(_args: &CpuKernelArgs) -> Result<(), CpuKernelLaunchError> {
        KERNEL_CALL_COUNT_V2.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn dispatcher_connects_to_registry_and_launches() {
        KERNEL_CALL_COUNT_V1.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig {
            cache_capacity: 1,
            main_memory_capacity: 1,
        });

        let fill_key = KernelKey::cpu(OpTag::Fill);
        let copy_key = KernelKey::cpu(OpTag::Copy);

        dispatcher
            .register(
                fill_key,
                KernelMetadata::cpu("fill_f32", test_fill_kernel_v1),
            )
            .expect("fill registration should succeed");
        dispatcher
            .register(
                copy_key,
                KernelMetadata::cpu("copy_f32", test_fill_kernel_v1),
            )
            .expect("copy registration should succeed");

        let args = KernelArgs::cpu(CpuKernelArgs::new());

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V1.load(Ordering::SeqCst), 1);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 1);

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should hit cache");
        assert_eq!(KERNEL_CALL_COUNT_V1.load(Ordering::SeqCst), 2);
        assert_eq!(dispatcher.registry().stats().cache_hits, 1);

        dispatcher
            .dispatch(copy_key, &args)
            .expect("copy dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V1.load(Ordering::SeqCst), 3);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 2);

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should rebuild from secondary");
        assert_eq!(KERNEL_CALL_COUNT_V1.load(Ordering::SeqCst), 4);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 3);
    }

    #[test]
    fn dispatcher_can_switch_to_v2_key_resolution() {
        KERNEL_CALL_COUNT_V2.store(0, Ordering::SeqCst);

        let mut dispatcher =
            Dispatcher::new(KernelRegistryConfig::default()).with_resolver_kind(ResolverKind::V2);
        let args = KernelArgs::cpu(CpuKernelArgs::new());

        let key = dispatcher.resolve_key(ExecutionTag::Cpu, OpTag::Fill, &args);
        assert_eq!(key.version(), KeyVersion::V2);

        dispatcher
            .register(key, KernelMetadata::cpu("fill_v2", test_fill_kernel_v2))
            .expect("v2 registration should succeed");

        dispatcher
            .dispatch_op(ExecutionTag::Cpu, OpTag::Fill, &args)
            .expect("v2 dispatch_op should succeed");
        assert_eq!(KERNEL_CALL_COUNT_V2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn kernel_key_v1_and_v2_roundtrip() {
        let v1 = KernelKey::v1(ExecutionTag::Cpu, OpTag::Fill);
        assert_eq!(v1.version(), KeyVersion::V1);
        assert!(v1.as_v1().is_some());
        assert!(v1.as_v2().is_none());
        assert_eq!(v1.execution(), ExecutionTag::Cpu);
        assert_eq!(v1.op(), OpTag::Fill);

        let v2 = KernelKey::v2(ExecutionTag::Cpu, OpTag::Copy, 0xDEAD_BEEF);
        assert_eq!(v2.version(), KeyVersion::V2);
        assert!(v2.as_v1().is_none());
        let decoded = v2
            .as_v2()
            .expect("v2 key should decode as v2 representation");
        assert_eq!(decoded.signature_hash(), 0xDEAD_BEEF);
        assert_eq!(decoded.execution(), ExecutionTag::Cpu);
        assert_eq!(decoded.op(), OpTag::Copy);
    }
}
