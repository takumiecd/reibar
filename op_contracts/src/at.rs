use crate::Scalar;

pub trait ReadAtOp<T> {
    type Error;
    fn read_at(&self, tensor: &T, indices: &[usize]) -> Result<Scalar, Self::Error>;
}

pub trait WriteAtOp<T> {
    type Error;
    fn write_at(&self, tensor: &T, indices: &[usize], value: Scalar) -> Result<(), Self::Error>;
}
