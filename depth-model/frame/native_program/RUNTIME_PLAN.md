# NPRG v2 native runtime placement plan

This is an offline placement plan, not an ownership or execution proof. The
NPRG v2 decoder tests prove metadata equivalence to QDRX; they do not prove that
the suggested addresses are free while Frame firmware is running.

## Measured split

The primary NPRG v2 image is 397,136 B (alternate 395,048 B), about half the
NPRG v1 size. Primary bytes are geometry 29,524; per-tensor storage links
64,032; delta block stream 161,666; commands 101,360; refs 25,946; framework
vectors 4,916; calls 9,416; plus header and alignment. Geometry is deduplicated
across 5,336 tensors into 671 exact immutable records. The 12-byte tensor row
contains a geometry index plus a tagged direct-pool location or compressed
block-stream offset/count/slice origin. It preserves each tensor's distinct
pool mapping, block list span and slice origin. The block stream stores each
identical ordered list once; each list starts with one 3-byte pool+offset value
and encodes subsequent same-pool signed offset deltas as ZigZag ULEB128.

For primary, the geometry/tensor/command/ref/vector/call sections and header
need 235,470 B hot. The 161,666 B compressed block stream can remain in the
private QDO heap and be expanded once during program open into 120,330 u32
entries (481,320 B). That setup cost is bounded and does not require per-command
DDR decode or per-command table copies. The native tensor descriptors point
directly at those expanded block arrays. The QDO heap is private memory from
`qd_owned_memory.h`/`QdsConfig`, separate from pool1. Do not put decoded block
tables in pool1 or the VTCM lease.

## Candidate L2 intervals

Nominal L2 is `[0x30000000, 0x30100000)` (`qd_conv.h`). The known existing
addresses suggest this candidate split after removing the old metadata cache:

| Candidate interval | Use | Capacity / evidence |
|---|---|---|
| `0x30000000..0x30080000` | existing scalar per-command scratch | 512 KiB; measured peak 491,968 B |
| `0x30080000..0x300b0000` | three 64 KiB worker stack slots and guards | `qd_graph_boot.c` passes stack tops `0x30090000 + tid*0x10000` |
| `0x300b0000..0x300ef000` | hot NPRG v2 metadata | 240 KiB; primary 235,470 B leaves 10,290 B |
| `0x300ef000..0x300f0000` | gap / map alignment | 4 KiB |
| `0x300f0000..0x30100000` | QdsWork plus conservative graph/worker state reserve | 64 KiB |

With those assumptions, primary hot metadata + the current scalar scratch peak
is 727,438 B under the 768 KiB planning ceiling (1 MiB less 192 KiB for the
three worker stack slots and 64 KiB state reserve), leaving 58,994 B. The
alternate leaves 60,134 B. Keeping the entire compressed NPRG v2 image hot
instead would require 889,104 B with current scratch, so it exceeds that
planning ceiling. The split is what makes the byte budget plausible.

The candidate hot interval currently holds `BlockCache` at `0x300b0000`, the
copied `QdgGraph` at `0x300b2000`, `TensorCache` at `0x300b4000`, and the
per-command metadata arena at `0x300b8000..0x300ef000`. All four must be retired
or relocated before assigning this interval to NPRG. The current convolution
worker uses the whole L2 TCM for packed weights/parameters; that worker cannot
run concurrently with this map and must be retired or have its scratch moved.
The first 512 KiB scratch arena also needs a command-liveness proof against
native worker concurrency.

These addresses are inferred from current code, not a linker-owned reservation
map. The 128-byte guard writes in `qd_graph_boot.c` do not measure stack high
water. Before runtime enablement, produce an exact ELF/linker memory map, prove
stack limits, snapshot the full L2 contents before and after an exclusive
session, and prove restore. Until then `l2_placement_proven` remains false.

## Pool1/VTCM ownership and no-fallback rule

Pool1 is the full 8 MiB window `[0x31000000,0x31800000)` while the exclusive
VTCM lease is held. Native open must bind pool1 to that base and validate the
QDRX pool size/hash against the selected model before any writes. Pool1 tensors
then read/write directly through their block descriptors; block-cache copies
must be bypassed. Existing HMX staging at `0x31000000` and legacy per-command
VTCM scratch cannot run beside those tensors. Native preflight is all-or-nothing:
if any command lacks a resolver or exceeds its declared per-command plus
concurrent-worker scratch budget, refuse before command 0. A native command
must never fall through to the old graph handler, `resident_request`, or a
VTCM scratch fallback while pool1 is mapped there.

## First runtime patch milestone

In a separate runtime change, add a small `nprg2_open`/`nprg2_bind` module and
one dispatch entry point. `open` verifies NPRG CRC, the trusted manifest's
prepared-image/source/context/constants hashes, ABI, all field narrowing and
every DDR/VTCM pool span; it expands the
block stream once into private QDO heap. `bind` copies only the 235,470 B hot
primary metadata into the candidate L2 interval, binds pool1 to the leased
VTCM base, and verifies map guards. The command resolver must inspect each
command's variant, tensor types/shapes/quantization/storage and scratch contract
and reject any legacy fallback. Start with the already numerically proven
1x1 feature set and its exact prepared command fixture; run no graph prefix
unless every command in that prefix has native coverage. Report a command/output
hash and exact L2/VTCM snapshot. Expand to 3x3 only through the existing separate
oracle proof set. This milestone is a decoder/dispatch gate, not an FPS or
20 ms claim.
