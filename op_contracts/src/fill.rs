pub trait FillOp {
    type Error;
    fn fill_inplace(&mut self, value: f32) -> Result<(), Self::Error>;
}
