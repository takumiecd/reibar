use execution::{KernelArgs, KernelLaunchError, KernelMetadata};

use crate::version::{V1KeyCodec, V1KeyParts, V2KeyCodec, V2KeyParts};
use crate::version::{VersionError, next_fallback_key};
use crate::{KernelKey, KernelRegistry, KernelRegistryStats, KeyError, KeyVersion, key_version};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    KeyVersion(KeyError),
    KeyVersionMismatch {
        dispatcher: KeyVersion,
        key: KeyVersion,
    },
    Version(VersionError),
    FallbackLoop {
        key: KernelKey,
    },
    FallbackOverflow {
        seed: KernelKey,
        max_steps: usize,
    },
    KernelNotFound {
        key: KernelKey,
    },
    Launch(KernelLaunchError),
}

#[derive(Debug, Clone, Copy)]
pub struct DispatchRequest<'a> {
    pub key: KernelKey,
    pub args: &'a KernelArgs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedSpec {
    V1(V1KeyParts),
    V2(V2KeyParts),
}

impl SeedSpec {
    pub fn version(&self) -> KeyVersion {
        match self {
            Self::V1(_) => KeyVersion::V1,
            Self::V2(_) => KeyVersion::V2,
        }
    }

    pub fn encode(&self) -> KernelKey {
        match self {
            Self::V1(parts) => V1KeyCodec::encode(*parts),
            Self::V2(parts) => V2KeyCodec::encode(*parts),
        }
    }
}

pub trait DispatchApi {
    fn dispatch(&mut self, request: DispatchRequest<'_>) -> Result<(), DispatchError>;
    fn dispatch_seed(&mut self, seed: SeedSpec, args: &KernelArgs) -> Result<(), DispatchError>;
}

#[derive(Debug, Clone)]
pub struct Dispatcher {
    registry: KernelRegistry,
    version: KeyVersion,
    max_fallback_steps: usize,
}

impl Dispatcher {
    pub fn new(config: crate::KernelRegistryConfig, version: KeyVersion) -> Self {
        Self {
            registry: KernelRegistry::new(config),
            version,
            max_fallback_steps: 16,
        }
    }

    pub fn version(&self) -> KeyVersion {
        self.version
    }

    pub fn set_version(&mut self, version: KeyVersion) {
        self.version = version;
    }

    pub fn with_max_fallback_steps(mut self, max_fallback_steps: usize) -> Self {
        self.max_fallback_steps = max_fallback_steps;
        self
    }

    pub fn max_fallback_steps(&self) -> usize {
        self.max_fallback_steps
    }

    pub fn register(
        &mut self,
        key: KernelKey,
        metadata: KernelMetadata,
    ) -> Result<(), DispatchError> {
        self.ensure_key_version(key)?;
        self.registry.register(key, metadata);
        Ok(())
    }

    pub fn initialize_builtins(&mut self) -> Result<(), DispatchError> {
        crate::kernels::register_builtins(self)
    }

    pub fn stats(&self) -> KernelRegistryStats {
        self.registry.stats()
    }

    fn ensure_key_version(&self, key: KernelKey) -> Result<(), DispatchError> {
        let key_version = key_version(key).map_err(DispatchError::KeyVersion)?;
        if key_version != self.version {
            return Err(DispatchError::KeyVersionMismatch {
                dispatcher: self.version,
                key: key_version,
            });
        }
        Ok(())
    }
}

impl DispatchApi for Dispatcher {
    fn dispatch(&mut self, request: DispatchRequest<'_>) -> Result<(), DispatchError> {
        self.ensure_key_version(request.key)?;
        let seed_key = request.key;
        let mut key = seed_key;
        let mut steps = 0usize;

        loop {
            if let Some(launcher) = self.registry.select(&key) {
                return launcher.launch(request.args).map_err(DispatchError::Launch);
            }

            if steps >= self.max_fallback_steps {
                return Err(DispatchError::FallbackOverflow {
                    seed: seed_key,
                    max_steps: self.max_fallback_steps,
                });
            }

            let next = next_fallback_key(self.version, key, request.args)
                .map_err(DispatchError::Version)?;

            match next {
                Some(next_key) => {
                    if next_key == key {
                        return Err(DispatchError::FallbackLoop { key });
                    }
                    key = next_key;
                    steps += 1;
                }
                None => return Err(DispatchError::KernelNotFound { key: seed_key }),
            }
        }
    }

    fn dispatch_seed(&mut self, seed: SeedSpec, args: &KernelArgs) -> Result<(), DispatchError> {
        let key = seed.encode();
        self.dispatch(DispatchRequest { key, args })
    }
}
