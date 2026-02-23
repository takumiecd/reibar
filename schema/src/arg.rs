use crate::{ArgKey, ArgValue, Scalar, ScalarBuffer};

#[derive(Debug, Clone, PartialEq)]
pub struct KernelArg<S> {
    key: ArgKey,
    value: ArgValue<S>,
}

impl<S> KernelArg<S> {
    pub fn new(key: ArgKey, value: ArgValue<S>) -> Self {
        Self { key, value }
    }

    pub fn storage(key: ArgKey, value: S) -> Self {
        Self::new(key, ArgValue::Storage(value))
    }

    pub fn scalar_buffer(key: ArgKey, value: ScalarBuffer) -> Self {
        Self::new(key, ArgValue::ScalarBuffer(value))
    }

    pub fn scalar(key: ArgKey, value: Scalar) -> Self {
        Self::new(key, ArgValue::Scalar(value))
    }

    pub fn f32(key: ArgKey, value: f32) -> Self {
        Self::scalar(key, Scalar::F32(value))
    }

    pub fn i64(key: ArgKey, value: i64) -> Self {
        Self::scalar(key, Scalar::I64(value))
    }

    pub fn u8(key: ArgKey, value: u8) -> Self {
        Self::scalar(key, Scalar::U8(value))
    }

    pub fn usize(key: ArgKey, value: usize) -> Self {
        Self::new(key, ArgValue::Usize(value))
    }

    pub fn bool(key: ArgKey, value: bool) -> Self {
        Self::scalar(key, Scalar::Bool(value))
    }

    pub fn key(&self) -> &ArgKey {
        &self.key
    }

    pub fn value(&self) -> &ArgValue<S> {
        &self.value
    }
}
