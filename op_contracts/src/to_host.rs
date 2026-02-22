pub trait ToHostOp<T> {
    type Error;
    fn to_host(&self, tensor: &T) -> Result<T, Self::Error>;
}
