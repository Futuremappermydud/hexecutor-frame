# Prepared validation plan cache prototype

Offline-only prototype notes for `NPRG2` schedule validation. No Frame device,
shared runtime, or `native_program/` file is changed here.

## What is safe to cache

An immutable plan can retain a handler registry slot (one byte per command) and
the session identity that made that slot valid. Creation must still run the
ordinary full preflight, including handler validation and all fail-closed range
checks. It can then store the resolved slot for each command. On reuse, the
session must confirm its plan token, interval, registry generation, pool base
and size identities, active pool-1 ceiling, and prepared image identity. The
execution path must still construct each call using the current request values
and execute handler-specific *dynamic* validation before any operation that
can write. A failed dynamic check aborts before the requested interval begins.

The current handlers do not expose static and dynamic validation separately.
Therefore a handler slot cache alone cannot safely skip `validate()`: doing so
would stale-check InputSlice IO pointers/lengths and LinearClip Fi values. The
prototype benchmark measures only the removable command decode/registry lookup
work; it does not claim the full proposed cache speedup.

## Required invariants

- Cache miss, identity mismatch, out-of-range command, absent tag, or failed
  static proof returns failure before pool writes. There is no legacy fallback.
- Dynamic IO pointer/length and pointer-overlap checks use the current request.
- Fi location is static metadata; Fi payload and any range predicate over it
  are checked against the current execution state. If an earlier command in
  the same range produces Fi, its predicate must be guaranteed for all values
  or validated without mutating graph memory before execution starts.
- Static shape, offsets, tensor references, allowed op/variant, block alias
  proof, and pool-1 writable-range proof may be retained only while all bound
  metadata and mappings are private and unchanged.
- The plan occupies 128-byte header + command tags rounded to 128 bytes. At
  5,068 / 5,040 commands this is 5,248 bytes. It must be placed after aligned
  `bound.hot_bytes`; current design margins are 17,606 / 18,746 bytes.

## Integration patch plan

1. Add explicit immutable-plan and per-execution-check callbacks to the
   executor registration API. Existing handlers default to uncached full
   validation; cache coverage is opt-in per handler.
2. Refactor handlers independently: static proof creates a compact immutable
   descriptor; dynamic check refreshes IO/Fi and other live-value predicates.
   Do not cache the descriptor until the callback pair is proven equivalent
   to the existing validator for both success and rejection cases.
3. Build runner-owned tags/descriptors only after ordinary full preflight.
   Bind plan identity to private runner session, exact interval, handler
   generation, pool mappings/limits, and prepared image hash. Place after hot
   metadata and reject if capacity is insufficient.
4. Keep request-wide alias checks and whole-range fail-closed rejection in the
   executor. Reuse plan only for the same interval; cancellation and execution
   outcomes stay per call.
5. Benchmark cache creation separately from repeated execution checks on
   primary and alternate NPRG2, including IO pointer changes, changed Fi, and
   invalid range rejection. Only report complete savings after the split
   handler API lands and those comparisons pass.

## Measured prototype scope

`benchmark_dispatch.py` decodes the actual primary-v2 and alternate-v2 NPRG2
images and compares repeated schedule decode + registry-probe simulation with
cached handler-tag access. It is a host microbenchmark of dispatch overhead,
not an NPRG2 executor benchmark, validation proof, full-model test, or Frame
latency result. Results are emitted as JSON for reproducibility.
