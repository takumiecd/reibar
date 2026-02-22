pub trait ReadAtF32Op<T> {
    type Error;
    fn read_at_f32(&self, tensor: &T, indices: &[usize]) -> Result<f32, Self::Error>;
}

pub trait WriteAtF32Op<T> {
    type Error;
    fn write_at_f32(&self, tensor: &T, indices: &[usize], value: f32) -> Result<(), Self::Error>;
}
