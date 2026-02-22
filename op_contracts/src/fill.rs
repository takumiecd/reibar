pub trait FillOp<T> {
    type Error;
    fn fill_inplace(&self, tensor: &mut T, value: f32) -> Result<(), Self::Error>;
}
