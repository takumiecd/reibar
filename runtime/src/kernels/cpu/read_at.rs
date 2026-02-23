use execution::{CpuKernelMetadata, ExecutionTag, KernelMetadata};

use crate::version::{V1KeyCodec, V1KeyParts};
use crate::{DispatchError, Dispatcher, OpTag};

pub fn register(dispatcher: &mut Dispatcher) -> Result<(), DispatchError> {
    let key = V1KeyCodec::encode(V1KeyParts {
        execution: ExecutionTag::Cpu,
        op: OpTag::ReadAt,
        selector: 0,
    });

    dispatcher.register(
        key,
        KernelMetadata::Cpu(CpuKernelMetadata::new(
            "cpu.read_at.v1.selector0",
            execution_cpu::kernel::cpu::read_at::launch,
        )),
    )
}
