pub mod at;
pub mod fill;
pub mod narrow;
pub mod to_host;

use op_contracts::{FillOp, NarrowOp, ReadAtF32Op, ToHostOp, WriteAtF32Op};

use crate::DenseTensorImpl;

pub struct DenseOps;

impl FillOp<DenseTensorImpl> for DenseOps {
    type Error = fill::DenseFillError;
    fn fill_inplace(&self, tensor: &mut DenseTensorImpl, value: f32) -> Result<(), Self::Error> {
        fill::exec(tensor, value)
    }
}

impl ReadAtF32Op<DenseTensorImpl> for DenseOps {
    type Error = at::DenseAtError;
    fn read_at_f32(&self, tensor: &DenseTensorImpl, indices: &[usize]) -> Result<f32, Self::Error> {
        at::exec_read(tensor, indices)
    }
}

impl WriteAtF32Op<DenseTensorImpl> for DenseOps {
    type Error = at::DenseAtError;
    fn write_at_f32(
        &self,
        tensor: &DenseTensorImpl,
        indices: &[usize],
        value: f32,
    ) -> Result<(), Self::Error> {
        at::exec_write(tensor, indices, value)
    }
}

impl NarrowOp<DenseTensorImpl> for DenseOps {
    type Error = narrow::DenseNarrowError;
    fn narrow(
        &self,
        tensor: &DenseTensorImpl,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<DenseTensorImpl, Self::Error> {
        narrow::exec(tensor, dim, start, len)
    }
}

impl ToHostOp<DenseTensorImpl> for DenseOps {
    type Error = to_host::DenseToHostError;
    fn to_host(&self, tensor: &DenseTensorImpl) -> Result<DenseTensorImpl, Self::Error> {
        to_host::exec(tensor)
    }
}
