# Prepared schedule static validation cache proposal

This is a design note only; the C runtime still performs ordinary full preflight.

## Cache layout and budget

Use a fixed 128-byte header followed by a bytecode-tag table sized to the full
prepared command count. Reserve one byte per schedule command: registry handler
index `0..79` for a statically covered command and `0xff` for uncovered. The
primary model has 5,068 commands and the alternate has 5,040, so rounding the
table allocation up to a 128-byte boundary takes 5,120 bytes in either model.
The total immutable plan reservation is therefore 5,248 bytes. The header can
hold plan magic/version, command count, requested interval, registry generation,
active pool-1 limit, seven pool base identities (28 bytes on Hexagon), and a
prepared-image/session fingerprint. Do not store function pointers per command
or page-address sets per operator.

The hot metadata arena is currently 0x3f000 bytes (258,048 bytes). The bound
tables consume 235,194 bytes in primary and 234,054 bytes in alternate, leaving
22,854 and 23,994 bytes respectively. The 5,248-byte plan fits with 17,606
bytes primary margin and 18,746 bytes alternate margin, provided
`bound.hot_bytes` is measured and the plan is placed only after its aligned
end. If hot metadata grows and that margin is insufficient, plan creation
must fail before any pool writes.
The existing 30,748-byte Hexagon elementwise alias scratch is setup-only once
static proof is cached; it can be reused for compilation but should remain a
caller-supplied bounded arena until the linker map explicitly assigns it. The
plan itself does not consume that workspace.

## Static proof to cache

At plan creation, walk each requested command in schedule order and resolve its
registered handler. Reject any missing handler, forbidden fallback, disabled
native family, invalid fixed shape/variant tuple, malformed pool/slice bounds,
reserved-tail write, or output/input/page alias before the executor writes to
any graph pool. For elementwise handlers, sort/compare block-address sets once
using the existing caller scratch; retain only the handler tag. For Im2col,
validate the active 32-page slice once; retain the tag and rely on the immutable
bound block table and stable pool base mapping. A 5,068-byte tag table is much
smaller than retaining per-command page lists (ADD/LinearClip proof sets could
otherwise take megabytes).

This removes repeated registry probing and repeated metadata decoding, and
lets the runner avoid the current second whole-range preflight after its own
pre-boundary proof. The plan must be private to the runner session; callers
must not receive writable aliases to bound hot records, expanded blocks, refs,
geometry, calls, the registry, or the plan bytes. Pool base pointers, pool
sizes, active pool-1 ceiling, prepared image, and registered handler set must
stay fixed for the plan lifetime.

## Dynamic fields that cannot be cached

- InputSlice caller IO data pointer/length can change each frame. Check the
  current pointer range and its overlap with the output before execution.
- LinearClip Fi payload values are live data. Re-read multiplier/shift from
  the current Fi location; do not cache their numerical values across frames.
  Its shape, pool, offset, and alias proof are static.
- Pool contents for weights and activation tensors remain dynamic even when
  their addresses and extents are static.
- Cancellation, requested range, and submit/drain outcome are per execution.

A tricky case is Fi produced earlier inside the same requested interval. A
setup-time check sees an old or zero Fi value. Before generalizing the cache,
handlers need separate APIs for static metadata proof and current-data checks.
Coverage and alias safety can be guaranteed for the full range before writes;
any dynamic value predicate that depends on an earlier command must either be
proven from that producer's static output domain, validated without mutation
before the range starts, or be defined for every possible produced value. Do
not silently treat the pre-run Fi bytes as authoritative.

## Remaining integration checks

The native convolution binder currently repeats feature/geometry checks and
writes its small tables/ABI on each command. A later plan may cache immutable
leaf descriptors while still publishing/cleaning the required 2 KiB tables,
120-byte ABI and tail before each HMX submission. That change needs a measured
barrier/cache-clean cost and the root's proven backend path. It is outside this
preprocess handler patch.
