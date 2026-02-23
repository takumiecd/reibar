#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EncodedScalar {
    len: u8,
    bytes: [u8; Self::MAX_BYTES],
}

impl EncodedScalar {
    pub const MAX_BYTES: usize = 8;

    pub fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len()]
    }

    pub(crate) fn from_array<const N: usize>(bytes: [u8; N]) -> Self {
        assert!(
            N <= Self::MAX_BYTES,
            "encoded scalar length exceeds maximum"
        );
        let mut out = [0u8; Self::MAX_BYTES];
        out[..N].copy_from_slice(&bytes);
        Self {
            len: N as u8,
            bytes: out,
        }
    }

    pub(crate) fn from_byte(byte: u8) -> Self {
        let mut out = [0u8; Self::MAX_BYTES];
        out[0] = byte;
        Self { len: 1, bytes: out }
    }
}

impl AsRef<[u8]> for EncodedScalar {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
