mod dispatcher;
mod key;
mod op;
mod registry;

pub use dispatcher::{DispatchError, Dispatcher};
pub use key::KernelKey;
pub use op::OpTag;
pub use registry::{
    KernelRegistry, KernelRegistryConfig, KernelRegistryError, KernelRegistryStats,
};

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use execution::{CpuKernelArgs, KernelArgs, KernelLaunchError, KernelMetadata};

    use super::{Dispatcher, KernelKey, KernelRegistryConfig, OpTag};

    static KERNEL_CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

    fn test_fill_kernel(_args: &CpuKernelArgs) -> Result<(), KernelLaunchError> {
        KERNEL_CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    #[test]
    fn dispatcher_connects_to_registry_and_launches() {
        KERNEL_CALL_COUNT.store(0, Ordering::SeqCst);

        let mut dispatcher = Dispatcher::new(KernelRegistryConfig {
            cache_capacity: 1,
            main_memory_capacity: 1,
        });

        let fill_key = KernelKey::cpu(OpTag::Fill);
        let copy_key = KernelKey::cpu(OpTag::Copy);

        dispatcher
            .register(fill_key, KernelMetadata::cpu("fill_f32", test_fill_kernel))
            .expect("fill registration should succeed");
        dispatcher
            .register(copy_key, KernelMetadata::cpu("copy_f32", test_fill_kernel))
            .expect("copy registration should succeed");

        let args = KernelArgs::cpu(CpuKernelArgs::new());

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 1);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 1);

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should hit cache");
        assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 2);
        assert_eq!(dispatcher.registry().stats().cache_hits, 1);

        dispatcher
            .dispatch(copy_key, &args)
            .expect("copy dispatch should succeed");
        assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 3);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 2);

        dispatcher
            .dispatch(fill_key, &args)
            .expect("fill dispatch should rebuild from secondary");
        assert_eq!(KERNEL_CALL_COUNT.load(Ordering::SeqCst), 4);
        assert_eq!(dispatcher.registry().stats().secondary_hits, 3);
    }
}
