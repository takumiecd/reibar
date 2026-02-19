#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpTag {
    Fill,
    Copy,
}

impl OpTag {
    pub(crate) fn code(self) -> u16 {
        match self {
            Self::Fill => 0,
            Self::Copy => 1,
        }
    }

    pub(crate) fn from_code(code: u16) -> Self {
        match code {
            0 => Self::Fill,
            1 => Self::Copy,
            _ => unreachable!("invalid op code generated internally: {code}"),
        }
    }
}
