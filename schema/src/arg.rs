use crate::{ArgKey, ArgValue, Scalar, ScalarBuffer, ViewSpec};

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

    pub fn usize(key: ArgKey, value: usize) -> Self {
        Self::new(key, ArgValue::Usize(value))
    }

    pub fn view_spec(key: ArgKey, value: ViewSpec) -> Self {
        Self::new(key, ArgValue::ViewSpec(value))
    }

    pub fn key(&self) -> &ArgKey {
        &self.key
    }

    pub fn value(&self) -> &ArgValue<S> {
        &self.value
    }
}
