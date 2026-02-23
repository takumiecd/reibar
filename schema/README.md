# schema

`schema` defines shared runtime vocabulary used across API/core/runtime/backend layers.

## Main Concepts

- `DType`: storage element type descriptor.
- `Scalar`: single runtime value.
- `ScalarBuffer`: typed vector payload (`dtype + len + bytes`).
- `ArgKey` / `ArgKind` / `KernelArg` / `KernelArgs`: kernel argument model.
- `ArgRole`: argument role (`Input`, `Output`, `Param`, ...).
- `KernelArgsError`: common validation errors.

## Design Notes

- `DType` is responsible for encoding/decoding policy and layout metadata.
- `Scalar` is responsible for concrete value payload.
- `KernelArgs` provides typed helpers:
  - `insert_scalar`
  - `require_scalar`
  - `require_scalar_bytes`
  - `insert_scalar_buffer`
  - `require_scalar_buffer`

## Reference

- Type-system design: `docs/type-system.md`
