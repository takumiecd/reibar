use crate::Scalar;

pub trait FillOp<T> {
    type Error;
    fn fill_inplace(&self, tensor: &mut T, value: Scalar) -> Result<(), Self::Error>;
}
