use crate::{ArgKey, ArgValue};

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

    pub fn f32(key: ArgKey, value: f32) -> Self {
        Self::new(key, ArgValue::F32(value))
    }

    pub fn i64(key: ArgKey, value: i64) -> Self {
        Self::new(key, ArgValue::I64(value))
    }

    pub fn u8(key: ArgKey, value: u8) -> Self {
        Self::new(key, ArgValue::U8(value))
    }

    pub fn usize(key: ArgKey, value: usize) -> Self {
        Self::new(key, ArgValue::Usize(value))
    }

    pub fn bool(key: ArgKey, value: bool) -> Self {
        Self::new(key, ArgValue::Bool(value))
    }

    pub fn key(&self) -> &ArgKey {
        &self.key
    }

    pub fn value(&self) -> &ArgValue<S> {
        &self.value
    }
}
