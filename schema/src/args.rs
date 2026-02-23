use crate::{ArgKey, ArgValueAccess, KernelArg, KernelArgsError};

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
        if arg.key().dtype() != arg.value().kind() {
            return Err(KernelArgsError::TypeMismatch {
                key: arg.key().clone(),
                expected: arg.key().dtype(),
                actual: arg.value().kind(),
            });
        }

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
