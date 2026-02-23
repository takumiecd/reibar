#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpTag {
    Fill,
    Copy,
    ReadAt,
    WriteAt,
}

impl OpTag {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Fill => 0,
            Self::Copy => 1,
            Self::ReadAt => 2,
            Self::WriteAt => 3,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Fill),
            1 => Some(Self::Copy),
            2 => Some(Self::ReadAt),
            3 => Some(Self::WriteAt),
            _ => None,
        }
    }
}
