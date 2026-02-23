use crate::{
    ArgKey, ArgValueAccess, DType, EncodedScalar, KernelArg, KernelArgsError, Scalar, ScalarBuffer,
    ViewSpec,
};

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

    pub fn insert_scalar(&mut self, key: ArgKey, value: Scalar) -> Result<(), KernelArgsError> {
        self.insert(KernelArg::scalar(key, value))
    }

    pub fn insert_scalar_buffer(
        &mut self,
        key: ArgKey,
        value: ScalarBuffer,
    ) -> Result<(), KernelArgsError> {
        self.insert(KernelArg::scalar_buffer(key, value))
    }

    pub fn require_scalar(&self, key: &ArgKey, dtype: DType) -> Result<Scalar, KernelArgsError> {
        match dtype {
            DType::F32 => self.require_as::<f32>(key).map(|v| Scalar::F32(*v)),
            DType::I64 => self.require_as::<i64>(key).map(|v| Scalar::I64(*v)),
            DType::U8 => self.require_as::<u8>(key).map(|v| Scalar::U8(*v)),
            DType::Bool => self.require_as::<bool>(key).map(|v| Scalar::Bool(*v)),
        }
    }

    pub fn require_encoded_scalar(
        &self,
        key: &ArgKey,
        dtype: DType,
    ) -> Result<EncodedScalar, KernelArgsError> {
        let value = self.require_scalar(key, dtype)?;
        match dtype.encode_scalar(&value) {
            Some(encoded) => Ok(encoded),
            None => Err(KernelArgsError::TypeMismatch {
                key: key.clone(),
                expected: dtype.value_arg_kind(),
                actual: value.arg_kind(),
            }),
        }
    }

    pub fn require_scalar_buffer(&self, key: &ArgKey) -> Result<&ScalarBuffer, KernelArgsError> {
        self.require_as::<ScalarBuffer>(key)
    }

    pub fn insert_view_spec(
        &mut self,
        key: ArgKey,
        value: ViewSpec,
    ) -> Result<(), KernelArgsError> {
        self.insert(KernelArg::view_spec(key, value))
    }

    pub fn require_view_spec(&self, key: &ArgKey) -> Result<&ViewSpec, KernelArgsError> {
        self.require_as::<ViewSpec>(key)
    }
}

impl<S> Default for KernelArgs<S> {
    fn default() -> Self {
        Self::new()
    }
}
