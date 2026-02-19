use crate::{ArgKey, ArgValueAccess, KernelArg, KernelArgsError, StorageValue};

#[derive(Debug, Clone, PartialEq)]
pub struct KernelArgs<S> {
    args: Vec<KernelArg<S>>,
}

impl<S> KernelArgs<S> {
    pub fn new() -> Self {
        Self { args: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.args.len()
    }

    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    pub fn insert(&mut self, arg: KernelArg<S>) -> Result<(), KernelArgsError> {
        if self.args.iter().any(|existing| existing.key() == arg.key()) {
            return Err(KernelArgsError::DuplicateKey {
                key: arg.key().clone(),
            });
        }

        self.args.push(arg);
        Ok(())
    }

    pub fn iter(&self) -> impl Iterator<Item = &KernelArg<S>> {
        self.args.iter()
    }

    pub fn require_storage(&self, key: &ArgKey) -> Result<&S, KernelArgsError> {
        self.require_as::<StorageValue>(key)
    }

    pub fn require_f32(&self, key: &ArgKey) -> Result<&f32, KernelArgsError> {
        self.require_as::<f32>(key)
    }

    pub fn require_i64(&self, key: &ArgKey) -> Result<&i64, KernelArgsError> {
        self.require_as::<i64>(key)
    }

    pub fn require_as<T>(&self, key: &ArgKey) -> Result<T::Ref<'_>, KernelArgsError>
    where
        T: ArgValueAccess<S>,
    {
        let arg = self.require_arg(key)?;
        match T::get(arg.value()) {
            Some(value) => Ok(value),
            None => Err(KernelArgsError::TypeMismatch {
                key: key.clone(),
                expected: T::KIND,
                actual: arg.value().kind(),
            }),
        }
    }

    fn require_arg(&self, key: &ArgKey) -> Result<&KernelArg<S>, KernelArgsError> {
        self.args
            .iter()
            .find(|arg| arg.key() == key)
            .ok_or_else(|| KernelArgsError::MissingKey { key: key.clone() })
    }
}

impl<S> Default for KernelArgs<S> {
    fn default() -> Self {
        Self::new()
    }
}
