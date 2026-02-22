use std::sync::{Mutex, OnceLock};

use execution::{CpuKernelArgs, KernelArgs, Storage};
use runtime::{
    DispatchApi, DispatchError, Dispatcher, KernelRegistryConfig, KeyVersion, OpTag, SeedSpec,
    V1KeyParts,
};
use schema::{ArgKey, ArgKind, ArgRole, KernelArg};

use crate::DenseTensorImpl;

pub fn exec(tensor: &mut DenseTensorImpl, value: f32) -> Result<(), DenseFillError> {
    if !tensor.is_packed() {
        return Err(DenseFillError::ViewNotSupported);
    }

    let mut cpu_args = CpuKernelArgs::new();
    let out = match tensor.storage() {
        Storage::Cpu(storage) => storage.clone(),
    };
    cpu_args
        .insert(KernelArg::storage(output_key(), out))
        .map_err(DenseFillError::KernelArgs)?;
    cpu_args
        .insert(KernelArg::f32(value_key(), value))
        .map_err(DenseFillError::KernelArgs)?;
    let args = KernelArgs::Cpu(cpu_args);

    let seed = SeedSpec::V1(V1KeyParts {
        execution: tensor.execution_tag(),
        op: OpTag::Fill,
        selector: 0,
    });

    let dispatcher = dense_dispatcher();
    let mut dispatcher = dispatcher
        .lock()
        .map_err(|_| DenseFillError::DispatcherPoisoned)?;
    dispatcher
        .dispatch_seed(seed, &args)
        .map_err(DenseFillError::Dispatch)?;

    Ok(())
}

#[derive(Debug)]
pub enum DenseFillError {
    ViewNotSupported,
    DispatcherPoisoned,
    KernelArgs(schema::KernelArgsError),
    Dispatch(DispatchError),
}

fn output_key() -> ArgKey {
    ArgKey::new(ArgRole::Output, "out", ArgKind::Storage)
}

fn value_key() -> ArgKey {
    ArgKey::new(ArgRole::Param, "value", ArgKind::F32)
}

fn dense_dispatcher() -> &'static Mutex<Dispatcher> {
    static DISPATCHER: OnceLock<Mutex<Dispatcher>> = OnceLock::new();
    DISPATCHER.get_or_init(|| {
        let mut dispatcher = Dispatcher::new(KernelRegistryConfig::default(), KeyVersion::V1);
        dispatcher
            .initialize_builtins()
            .expect("dense builtins initialization must succeed");
        Mutex::new(dispatcher)
    })
}
