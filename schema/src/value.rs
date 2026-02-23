#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgKind {
    Storage,
    F32,
    I64,
    U8,
    Usize,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StorageValue;

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue<S> {
    Storage(S),
    F32(f32),
    I64(i64),
    U8(u8),
    Usize(usize),
    Bool(bool),
}

impl<S> ArgValue<S> {
    pub fn kind(&self) -> ArgKind {
        match self {
            Self::Storage(_) => ArgKind::Storage,
            Self::F32(_) => ArgKind::F32,
            Self::I64(_) => ArgKind::I64,
            Self::U8(_) => ArgKind::U8,
            Self::Usize(_) => ArgKind::Usize,
            Self::Bool(_) => ArgKind::Bool,
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

impl<S> ArgValueAccess<S> for f32 {
    type Ref<'a>
        = &'a f32
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::F32;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::F32(v) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for i64 {
    type Ref<'a>
        = &'a i64
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::I64;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::I64(v) => Some(v),
            _ => None,
        }
    }
}

impl<S> ArgValueAccess<S> for u8 {
    type Ref<'a>
        = &'a u8
    where
        S: 'a;

    const KIND: ArgKind = ArgKind::U8;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::U8(v) => Some(v),
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

    const KIND: ArgKind = ArgKind::Bool;

    fn get<'a>(value: &'a ArgValue<S>) -> Option<Self::Ref<'a>>
    where
        S: 'a,
    {
        match value {
            ArgValue::Bool(v) => Some(v),
            _ => None,
        }
    }
}
