use crate::{ArgKind, ArgRole};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArgTag(String);

impl ArgTag {
    pub fn new(tag: impl Into<String>) -> Self {
        Self(tag.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ArgTag {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ArgTag {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArgKey {
    role: ArgRole,
    tag: ArgTag,
    dtype: ArgKind,
}

impl ArgKey {
    pub fn new(role: ArgRole, tag: impl Into<ArgTag>, dtype: ArgKind) -> Self {
        Self {
            role,
            tag: tag.into(),
            dtype,
        }
    }

    pub fn role(&self) -> ArgRole {
        self.role
    }

    pub fn tag(&self) -> &ArgTag {
        &self.tag
    }

    pub fn dtype(&self) -> ArgKind {
        self.dtype
    }
}
