use schema::DType;

/// A storage handle for a backend execution.
///
/// # Clone contract
///
/// `Clone::clone` MUST produce a **shared handle** to the same underlying
/// allocation (shallow/Arc-like copy). Writing through one clone MUST be
/// visible when reading through any other clone.
///
/// Deep-copy semantics are explicitly prohibited. Kernel dispatch code passes
/// a cloned storage handle to a kernel and expects mutations to be reflected
/// in the original tensor's storage. If cloning produces an independent
/// allocation, writes will be silently lost.
///
/// Implementors SHOULD verify this invariant with a unit test:
/// ```ignore
/// let a = MyStorage::allocate(...);
/// let b = a.clone();
/// write_value_through(&b, 42);
/// assert_eq!(read_value_through(&a), 42);
/// ```
pub trait ExecutionStorage: Clone + Send + Sync + 'static {
    fn len_bytes(&self) -> usize;
    fn dtype(&self) -> DType;
}

pub trait ExecutionStorageContext: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelContext: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelArgs: Clone + Send + Sync + 'static {}

pub trait ExecutionCapability: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelMetadata: Clone + Send + Sync + 'static {
    type Launcher;

    fn into_launcher(self) -> Self::Launcher;
}

/// A launch handle for a backend kernel implementation.
///
/// # Clone contract
///
/// `Clone::clone` SHOULD be cheap. Backends that manage expensive runtime
/// objects (for example, compiled device functions or pipeline handles)
/// SHOULD implement clone as a shared-handle copy (Arc-like), not a deep copy.
///
/// `to_metadata` SHOULD produce a compact descriptor suitable for caching.
/// Heavy runtime objects should remain owned by the launcher side.
pub trait ExecutionKernelLauncher: Clone + Send + Sync + 'static {
    type Metadata;
    type Args;
    type Error;

    fn to_metadata(&self) -> Self::Metadata;
    fn launch(&self, args: &Self::Args) -> Result<(), Self::Error>;
}
