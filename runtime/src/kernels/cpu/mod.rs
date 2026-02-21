mod fill;

use crate::{DispatchError, Dispatcher};

pub fn register(dispatcher: &mut Dispatcher) -> Result<(), DispatchError> {
    fill::register(dispatcher)
}
