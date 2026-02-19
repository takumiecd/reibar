# runtime

`runtime` owns dispatch orchestration (`dispatcher`, `kernel registry`, key resolution).

## Key Versioning Policy

- `KeyVersion::V1`:
  - Stable minimal key (`execution + op`).
  - Default for regular flows.
- `KeyVersion::V2`:
  - Implemented, but positioned as a reserved expansion track for richer matching.
  - Intended for future signature-oriented routing (`dtype`, layout, shape class, context traits).
  - Use only via explicit opt-in (`ResolverKind::V2`).

This keeps migration controlled while allowing key schema evolution without breaking existing registrations.
