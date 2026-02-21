pub trait ExecutionStorage: Clone + Send + Sync + 'static {
    fn len_bytes(&self) -> usize;
}

pub trait ExecutionStorageContext: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelContext: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelArgs: Clone + Send + Sync + 'static {}

pub trait ExecutionCapability: Clone + Send + Sync + 'static {}

pub trait ExecutionKernelMetadata: Clone + Send + Sync + 'static {
    type Launcher;

    fn into_launcher(self) -> Self::Launcher;
}

pub trait ExecutionKernelLauncher: Clone + Send + Sync + 'static {
    type Metadata;
    type Args;
    type Error;

    fn to_metadata(&self) -> Self::Metadata;
    fn launch(&self, args: &Self::Args) -> Result<(), Self::Error>;
}
