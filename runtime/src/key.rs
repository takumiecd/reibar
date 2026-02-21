pub type KernelKey = u64;

pub const KEY_VERSION_SHIFT: u32 = 60;
pub const KEY_VERSION_BITS: u32 = 4;
pub const KEY_VERSION_MASK: u64 = ((1u64 << KEY_VERSION_BITS) - 1) << KEY_VERSION_SHIFT;
pub const KEY_PAYLOAD_MASK: u64 = !KEY_VERSION_MASK;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyVersion {
    V1 = 1,
    V2 = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyError {
    UnknownVersionBits(u8),
}

pub fn compose_key(version: KeyVersion, payload: u64) -> KernelKey {
    ((version as u64) << KEY_VERSION_SHIFT) | (payload & KEY_PAYLOAD_MASK)
}

pub fn key_payload(key: KernelKey) -> u64 {
    key & KEY_PAYLOAD_MASK
}

pub fn key_version_bits(key: KernelKey) -> u8 {
    ((key & KEY_VERSION_MASK) >> KEY_VERSION_SHIFT) as u8
}

pub fn key_version(key: KernelKey) -> Result<KeyVersion, KeyError> {
    match key_version_bits(key) {
        1 => Ok(KeyVersion::V1),
        2 => Ok(KeyVersion::V2),
        bits => Err(KeyError::UnknownVersionBits(bits)),
    }
}
