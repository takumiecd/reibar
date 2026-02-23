# reibar

`reibar` is a Rust project to use different backends through a unified interface.

The name is inspired by *reinforcing bar* ("rebar"), with the project-specific spelling `reibar`.

## Goal

- Provide a consistent way to use and switch multiple Rust backends.
- Keep application-side usage stable even if backend implementations differ.

## Status

Work in progress.

## Design Docs

- Type-system guide (`DType`, `Scalar`, `Arg*`): `docs/type-system.md`
- Schema module guide: `schema/README.md`

## Local CI

Run the same `check-and-test` job as GitHub Actions with one command:

```bash
make ci
```

## Layers

- `api/`: user-facing API layer
- `core/`: concrete tensor implementations
- `execution/`: runtime abstraction layer (design notes first)
- `execution_cpu/`: CPU backend components (storage/kernel/context/capability bundle)
- `runtime/`: dispatcher and kernel registry orchestration
