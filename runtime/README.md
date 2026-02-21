# runtime

`runtime` owns dispatch orchestration (`dispatcher`, `kernel registry`, key resolution).

## Dispatcher Interface

- Stable request surface for ops:
  - `DispatchRequest { key, args }`
- Key creation and key lookup are separated:
  - key creation: ops/resolver/codec side
  - key lookup: dispatcher/registry side
- Dispatcher explicitly declares supported key version (`V1` or `V2`).
- Dispatch fallback is incremental:
  - try current key
  - ask version fallback policy for `next(current)`
  - repeat until hit or exhausted

## Key Versioning Policy

- `KeyVersion::V1`:
  - Stable minimal track.
- `KeyVersion::V2`:
  - Implemented with different payload layout.
  - Intended for future signature-oriented routing (`dtype`, layout, shape class, context traits).
  - Use via explicit dispatcher version selection.

This keeps migration controlled while allowing key schema evolution without breaking existing registrations.

## Layout

- `version/api.rs`: shared traits
- `version/v1/{codec,resolver,fallback}.rs`
- `version/v2/{codec,resolver,fallback}.rs`
