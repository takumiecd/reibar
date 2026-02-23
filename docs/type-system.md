# Type System Guide (`DType`, `Scalar`, `Arg*`)

This document defines responsibilities and boundaries for runtime types.

## Goal

- Keep memory representation stable.
- Keep kernel argument handling simple.
- Avoid mixing "data value", "data type", and "index semantics".

## Core Roles

### `DType`

`DType` is a type descriptor for storage layout and encoding rules.

- Owns: `size_bytes`, `alignment`, arg-kind mapping for scalar params.
- Owns: scalar encode/decode policy (`encode_scalar`, `decode_scalar`).
- Does not own: concrete runtime values.

### `Scalar`

`Scalar` is a runtime single value.

- Owns: value payload (`F32`, `I64`, `U8`, `Bool`).
- Owns: runtime value metadata (`dtype`, `arg_kind`).
- Does not own: buffer layout policy (delegated to `DType`).

### `ScalarBuffer`

`ScalarBuffer` is a typed vector payload: `{ dtype, len, bytes }`.

- Owns: contiguous vector bytes plus element type/length invariants.
- Owns: scalar decode at index (`scalar_at`).
- Does not own: backend memory handle (that remains `Storage` responsibility).

### `Storage`

`Storage` is bulk data (`bytes + dtype + backend handle`).

- Owns: memory region and backend handle.
- Does not own: per-element semantic meaning.

### `ArgKey` / `ArgKind` / `KernelArgs`

Kernel argument layer.

- `ArgKey`: semantic slot identity (`role + tag + expected kind`).
- `ArgKind`: transport kind check.
- `KernelArgs`: insertion/type validation and typed extraction.

`KernelArgs` convenience API should be preferred over ad-hoc per-kernel parsing:

- `insert_scalar`
- `require_scalar`
- `require_scalar_bytes`
- `insert_scalar_buffer`
- `require_scalar_buffer`

## Why `DType` and `Scalar` both exist

They are intentionally different dimensions:

- `DType` answers: "How should bytes be interpreted?"
- `Scalar` answers: "What value do we currently hold?"

They overlap only at conversion boundaries, which is expected.

## Index vs Scalar

`usize`-based index arguments are not just numeric values; they represent positions.

Recommendation:

- Keep index-like parameters distinct from scalar data parameters.
- Do not collapse all numbers into `Scalar` if operation semantics differ.

This keeps APIs readable and prevents accidental misuse.

## Recommended `ArgKind` direction

Current model is valid:

- `Storage`, `ScalarBuffer`, `Scalar(DType)`, `Usize`

Long-term simplification can be considered, but only with explicit index handling:

- Candidate target: `Storage`, `Scalar(DType)`, `Index`

Avoid a 2-kind model (`Storage` + `Scalar`) unless index semantics are encoded separately.

## Canonical Conversion Flow

1. API receives value (`impl Into<Scalar>`).
2. Tensor op checks `tensor.dtype()` and `value.dtype()` compatibility.
3. Use `KernelArgs::insert_scalar` for scalar params.
4. Kernel side uses `require_scalar` or `require_scalar_bytes`.
5. For byte conversion, always delegate to `DType::{encode_scalar, decode_scalar}`.

## Adding New Scalar Type Checklist

When adding a new scalar type (example: `I32`):

1. Add `DType::I32`.
2. Add `Scalar::I32`.
3. Extend `ArgKind`/`ArgValue`/`KernelArg` and typed access.
4. Extend `DType::size_bytes`, `alignment`, `value_arg_kind`, encode/decode.
5. Extend resolver mapping where `ArgKind` is fingerprinted.
6. Add schema tests for insert/extract/encode/decode.
7. Add kernel + op tests for read/write/fill paths.

## Current Decision (project baseline)

- Keep `DType` and `Scalar` separate.
- Keep index parameters separate from scalar value parameters.
- Use `ScalarBuffer` for typed vector params (`Vec`-like inputs) instead of ad-hoc byte blobs.
- Centralize scalar encoding/decoding and arg-kind mapping inside `schema`.
- Keep kernel code thin by using `KernelArgs` scalar helpers.
