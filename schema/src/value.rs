use crate::{DType, Scalar, ScalarBuffer, ViewSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgKind {
    Storage,
    ScalarBuffer,
    Scalar(DType),
    Usize,
    ViewSpec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StorageValue;

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue<S> {
    Storage(S),
    ScalarBuffer(ScalarBuffer),
    Scalar(Scalar),
    Usize(usize),
    ViewSpec(ViewSpec),
}

impl<S> ArgValue<S> {
    pub fn kind(&self) -> ArgKind {
        match self {
            Self::Storage(_) => ArgKind::Storage,
            Self::ScalarBuffer(_) => ArgKind::ScalarBuffer,
            Self::Scalar(v) => v.arg_kind(),
            Self::Usize(_) => ArgKind::Usize,
            Self::ViewSpec(_) => ArgKind::ViewSpec,
        }
    }
}

pub trait ArgValueAccess<S> {
    type Ref<'a>
    where
        S: 'a;

    const KIND: ArgKind;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a;
}

impl<S> ArgValueAccess<S> for StorageValue {
    type Ref<'a>
        = &'a S
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Storage;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Storage(storage) => Some(storage),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for ScalarBuffer {
    type Ref<'a>
        = &'a ScalarBuffer
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::ScalarBuffer;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::ScalarBuffer(v) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for f32 {
    type Ref<'a>
        = &'a f32
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Scalar(DType::F32);

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Scalar(Scalar::F32(v)) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for i64 {
    type Ref<'a>
        = &'a i64
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Scalar(DType::I64);

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Scalar(Scalar::I64(v)) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for u8 {
    type Ref<'a>
        = &'a u8
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Scalar(DType::U8);

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Scalar(Scalar::U8(v)) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for usize {
    type Ref<'a>
        = &'a usize
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Usize;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Usize(v) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for bool {
    type Ref<'a>
        = &'a bool
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::Scalar(DType::Bool);

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Scalar(Scalar::Bool(v)) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for ViewSpec {
    type Ref<'a>
        = &'a ViewSpec
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::ViewSpec;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::ViewSpec(v) => Some(v),
            _ => None,
        }
    }
}
