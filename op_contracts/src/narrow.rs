pub trait NarrowOp<T> {
    type Error;
    fn narrow(&self, tensor: &T, dim: usize, start: usize, len: usize) -> Result<T, Self::Error>;
}
