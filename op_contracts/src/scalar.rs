#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    F32(f32),
    I64(i64),
    Bool(bool),
}
