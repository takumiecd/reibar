mod fill;
mod read_at;
mod write_at;

use crate::{DispatchError, Dispatcher};

pub fn register(dispatcher: &mut Dispatcher) -> Result<(), DispatchError> {
    fill::register(dispatcher)?;
    read_at::register(dispatcher)?;
    write_at::register(dispatcher)
}
