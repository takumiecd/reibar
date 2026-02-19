# execution

`execution` is the runtime abstraction layer.

## Purpose

- Absorb backend-specific behavior behind one execution-facing API.
- Provide primitives that upper layers (for example `api/tensor`) can build on.

## Current design notes

- Keep user-facing tags stable (for example `Dense`) even if internal implementations change.
- Use `enum + tag` at the `execution` boundary.
- Hold backend/runtime-specific values in execution-side structs (for example storage handles).
- Use type erasure in this layer to hide concrete backend types from upper layers.

## Scope now

- CPU execution and shared storage are implemented as the first concrete runtime pieces.
- No `backends` layer yet.
