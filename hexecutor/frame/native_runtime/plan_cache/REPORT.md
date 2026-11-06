# NPRG2 validation plan cache prototype report

## Result

This offline prototype confirms that a one-byte tag per command is enough to
retain handler-slot resolution for the current serialized schedules. Actual v2
images contain 5,068 primary and 5,040 alternate commands. A rounded 128-byte
tag table is 5,120 bytes; with a 128-byte header the plan reservation is 5,248
bytes. The model schedule has 76 distinct `(opcode, variant)` signatures.

`dispatch_benchmark.json` records 500 host Python rounds. The benchmark decodes
the real 20-byte NPRG2 command records and repeats the executor's 80-entry
open-addressed hash lookup against a synthetic registry populated from the
image's signatures. It compares this with reading a prebuilt byte-tag list:

| Image | Decode + hash probe median / pass | Cached tag walk median / pass | Proxy delta |
|---|---:|---:|---:|
| Primary | 1,932.23 µs | 69.90 µs | 1,862.33 µs |
| Alternate | 1,944.63 µs | 64.21 µs | 1,880.42 µs |

These figures are **not a runtime speedup measurement**. This Python host proxy
executes no `nprg2_command_at`, handler validator, tensor/block alias proof,
pool range validation, IO validation, Fi read, executor, DSP instruction, or
Frame operation. The synthetic registry also does not establish prepared
handler coverage. It isolates only the shape and approximate overhead of
repeated command decode plus dispatch resolution in this proxy. It cannot
predict how much of whole-range preflight time a C/Hexagon implementation
would save.

## Safety and handler API requirement

The current executor calls a single `validate(call, ctx)` callback for every
command during each preflight. Existing handlers mix immutable checks with
execution-varying checks. For example:

- InputSlicePad derives static tensor/shape and destination bounds, then reads
  the current `inputs[io_index].data` and `.bytes` and proves current pointer
  range and source/output non-overlap.
- LinearClip resolves static blocked tensor descriptors and Fi location, but
  `linearclip_params` reads current Fi multiplier/shift values. If Fi is
  produced earlier in the requested interval, setup-time bytes are stale and
  cannot authorize a later operation.

Skipping these validators would permit stale IO/Fi data or stale pointer
aliases. To safely remove repeated static work, each cacheable handler needs
separate APIs, conceptually:

1. `plan_static(call, descriptor_out)`: prove handler support, fixed opcode /
   variant, tensor/reference geometry, immutable pool/offset/slice bounds,
   and static alias properties; retain only a small descriptor/tag.
2. `check_dynamic(call, descriptor)`: on every execution, re-read current IO
   pointer/length or Fi values and check all dynamic ranges, overlap, and value
   predicates. Reject before any graph write if current checks fail.
3. `run(call, descriptor)`: execute only after the entire requested range has
   passed static plan validation and dynamic checks.

If dynamic Fi validity depends on an earlier command in the same interval,
either prove the producer's output domain guarantees validity, validate that
predicate without writes before execution, or reject cached execution for that
handler. Never accept old Fi contents as proof.

## Integration plan

1. Keep current full preflight as the cache compiler's source of truth. Build
   the private plan only after it succeeds; missing/unsupported handlers or any
   bad range still fail closed before pool writes.
2. Give registrations explicit static-plan/dynamic-check callbacks and a
   generation counter. Existing callbacks remain uncached until split and
   parity tests prove accepted and rejected outcomes match the old validator.
3. Store tags/descriptors in the runner-owned hot arena immediately after
   aligned `bound.hot_bytes`; reject plan creation if the 5,248-byte reserve
   does not fit. Current documented remaining margin is 17,606 bytes primary
   and 18,746 bytes alternate.
4. Bind the plan to the exact runner session, prepared image identity,
   registry generation, requested interval, pool base/size identities, and
   active pool-1 ceiling. Recheck current request-wide aliases, IO pointers,
   Fi values, cancellation, and submit/drain outcomes each execution.
5. Only after callback splitting, benchmark C host and target preflight paths
   for both models. Include changed IO pointers/lengths, changed and
   in-interval-produced Fi, deliberately invalid ranges, and cache identity
   mismatch. Compare validator results and prove all rejections precede writes.

No production cache/runtime patch is included in this lane. No whole-model
coverage, full-model execution, or Frame speedup is claimed.
