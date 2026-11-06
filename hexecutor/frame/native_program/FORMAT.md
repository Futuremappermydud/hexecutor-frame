# Prepared native program prototype

`NPRG` is an offline-lowered execution table derived from QDRX v1. It removes
QDRX padding/debug fields, narrows safe references and block addresses, and
keeps a model binding to the source QDRX and its context/constants hashes. It
does not contain kernels or constitute a runnable interpreter.

All sections are little-endian, 8-byte aligned, canonical, with zero padding.
The 240-byte header stores version, total size, CRC32 (CRC field zeroed), QDRX
ABI, counts, SHA256(context), SHA256(constants), SHA256(source QDRX), and seven
offset/byte-length pairs in this order: tensors, blocks, commands, refs,
framework vectors, calls, graph I/O refs. It also stores all seven QDRX pool
sizes for compact-pointer validation.

Tensor records are 52 bytes. They store dtype/class/flags; a 24-byte union for
12 signed-16 shape/max-shape/padding values, four signed-32 shape values, or a
scalar word; direct pool/offset OR block-table first/count/slice; and
zero-point/scale. Shape narrowing is accepted only after range validation.
Block entries are 3 bytes: pool in bits 21..23 and byte offset / 128 in bits
0..20. The builder checks 128-byte alignment and pool bounds. It retains each
tensor's original block span and slice offset; it does not merge view records
or infer uniqueness from an equal block list.

Command records are 20 bytes in schedule order: u16 source sequence, u8 opcode,
u8 variant, u16 I/O index, u16 input-ref first, u16 output-ref first, u16
framework-vector first, u16 calls first, u8 input count, u8 output count, u8
framework-vector count, u8 call count, u8 duplicate-pad flag, and one zero byte.
Tensor refs and graph I/O refs are u16 indices. Framework checkpoint/word
arrays, `after` dependency numbers, and original producer IDs are validation
or audit fields; they are not consumed by the current serial graph dispatcher.
Call records remain 88 bytes because current operation handlers inspect their
IDs, masks and ten 64-bit arguments. Dynamic Fi/weight values remain ordinary
runtime tensors addressed by the retained tensor storage descriptors and
ordered commands; no activation or dynamic value is snapshotted here.

The Python decoder validates bounds, checksums, hashes, canonical offsets,
field narrowing, tensor refs, and command order. `preflight_native_program` is
the mandatory fail-closed gate for a future native mode: every scheduled
command needs a feature-aware resolver and declared scratch envelope.

## Minimal interpreter shape

At session setup, validate NPRG completely, check context/constants/source-image
hashes against the selected package, bind each declared pool to the session's
verified DDR/VTCM mapping, then copy the complete NPRG into the reserved L2
program arena once. Run the opcode-coverage preflight before starting workers
or executing any command. The command loop can then read each 20-byte command
row in-place; input/output refs index the compact tensor array directly, and
tensor block spans point into the prepared block array. This replaces
`dsp_command_begin` and its per-command copies, `dsp_tensor`'s linear search
through up to 32 entries, and temporary block-table copies. Keep the command
array order as the serial execution order; the source `sequence` is retained
for diagnostics but is not a sort key.

`DIRECT` and `BLOCKED` are locations, not cached values. Weights and Fi tensors
still resolve through their pool/offset or pool/block-span on each operation;
the command order still places their writes before consumers. Do not convert
pointer-placeholder call arguments to addresses. Pool pointer binding and
scratch/lifetime checks remain session-level obligations. This representation
does not remove any operation implementation requirement: with a partial
kernel set, preflight refuses the entire model before the first command.

Runtime-required fields are tensor dtype/class/flags, shape or shape-value or
scalar payload, storage location, quantization, command opcode/variant/order,
I/O index, input/output refs, framework vectors, calls, graph I/O refs and pool
sizes. QDRX command dependency number, checkpoints, framework words, and
original debug/provenance identifiers are retained in the source QDRX for
offline audit but omitted from this serial runtime table. The source image SHA
binds those omitted fields as well, so a prepared table cannot silently be
paired with a different schedule.

## Per-command native coverage and scratch contract

Opcode matching alone cannot authorize a command. `preflight_native_program`
requires an explicit resolver for every command; the resolver receives the
full command features: opcode and variant, each input/output tensor's type,
class, shape/max-shape/padding, quantization, storage pools/block spans/slice
origin, framework vector, call IDs/masks/arguments, and pool sizes. A resolver
must check all features relevant to that kernel's numerical proof. The
`exact_feature_capability` helper is a conservative starting point: it accepts
only explicitly approved full-feature fingerprints. No resolver is registered
by default, so the default result is reject-all.

Every accepted command must provide a `Capability` with its maximum private
scratch, any scratch live concurrently in workers, and `may_fallback=False`.
The preflight rejects the whole model before execution if either scratch peak
exceeds its supplied budget or a capability permits legacy fallback. A runtime
using pool1 as VTCM must dispatch only to the exact native implementation chosen
by the resolver; it must never fall through to `qd_graph.c`/`resident_request`
scratch paths. Missing capability, shape, variant, quantization, alignment, or
worker evidence refuses native mode before command 0. The Python API models
this gate but no DSP launch path is wired here.

The 1x1 adapter's supported prefix and the 3x3 Xwb work are separate proof
sets. One `k_hmx_conv` opcode covers multiple variants, dtypes, input/output
layouts, kernel sizes, strides, quantization modes and parameter layouts. A
proof for one such feature tuple must not enable the other variants or shapes.

## L2 placement and scratch budget (planning only)

NPRG v1 is 801,704 / 801,832 B. It does not fit the corrected stack-aware
planning ceiling: nominal L2 is 1 MiB, the three worker stack slots use 192 KiB
by their `0x10000` stride, and a provisional 64 KiB is reserved for graph/worker
state, leaving 768 KiB. The current generic scalar-kernel scratch peak is
491,968 B (`build/resident-gap-E.json`). See [RUNTIME_PLAN.md](RUNTIME_PLAN.md)
for the measured NPRG v2 hot/cold split and candidate address intervals.

Even v2's byte budget is not a placement proof. The old convolution worker
uses the full L2 TCM base for weight/parameter staging (`qd_conv.h` and
`qd_conv.c`); the old block cache, copied graph, tensor cache and copied
metadata arena occupy the candidate hot interval. These must be retired or
relocated. `qd_graph_boot.c` guards only 128 B per worker, so stack high-water
is unknown. A linker map and before/after preservation proof for the full L2
range are required before assigning those addresses.

Every command capability must declare local and concurrent worker scratch.
NPRG v2's hot primary metadata plus the measured old scalar scratch peak totals
727,438 B under the 768 KiB planning ceiling, but the remaining 58,994 B is
only a planning margin. The current full convolution worker cannot share this
map. Any native path with pool1 bound to the full 8 MiB VTCM must refuse before
execution when an operation is unsupported; legacy VTCM scratch fallback is
never permitted.

Metadata traffic estimate is the bytes copied from the uncached QDRX image by
the current per-command path (command record, input/output refs, framework
vectors, calls, and each distinct tensor descriptor/block list lazily touched),
versus one-time prepared-image upload/copy. It excludes graph tensor traffic,
kernel input packing, weights, and pool transfers. It is an upper-bound style
structural estimate, not a latency prediction.
