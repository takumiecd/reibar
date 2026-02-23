mod cpu;

use crate::{DispatchError, Dispatcher};

pub fn register_builtins(dispatcher: &mut Dispatcher) -> Result<(), DispatchError> {
    cpu::register(dispatcher)
}
