# NPRG v2 prepared schedule executor

`prepared_executor.c/.h` adds a small serial control loop over the bound NPRG v2
tables. The fixed registry holds at most 80 `(opcode, variant)` handler
records; it is not indexed by command and does not allocate a per-command
dispatch pointer table. A handler has separate validation and execution
callbacks.

`nprg2_executor_run` performs two passes over the requested command interval.
The first resolves every command key and runs every handler validator. A
missing key or failed predicate rejects the interval before any handler runs.
The second pass calls handlers in stored schedule order, supports cancellation
between commands, and stops on the first handler error. The module has no
legacy dispatcher or fallback hook. Validators must check feature coverage,
tensor geometry/storage and their scratch requirement using the supplied
call's scratch pointer/size. Pool mappings are checked against the mappings
used by `nprg2_open`; command refs and tensor views come only from the bound
hot tables. Blocked views use the expanded block table created once by open.
Handlers can declare private and concurrent-worker scratch bytes at registry
registration; preflight sums those requirements and compares them with the
single caller-provided scratch arena before calling the feature validator.
Validators must be read-only. The executor also rejects scratch, stats, IO,
registry, and native table/ABI buffers that overlap graph pools or bound/raw
program metadata.

The built-in `QDR_WEIGHTS_TO_VTCM` handler supports only variants 27 through
30, one direct input and output, the matching lower/upper class pair, equal
data types, shape geometry, and destination pool 1. It bounds both addresses
against the supplied pool sizes and uses `memmove` for the exact old-kernel
extent: the smaller max-shape byte count. It allocates no scratch and makes no
assumption that source/destination overlap is impossible. The handler tests
against CPU byte-copy semantics; it does not prove a Qualcomm VTCM mapping.

The host test runs a mock no-op registry through both actual model schedules to
prove registry coverage, order/count and fail-before-write behavior. It then
executes actual `WEIGHTS_TO_VTCM` command metadata for variants 27–30 in each
model with mock pools and checks the copied bytes. Mock schedule execution makes no
numerical or kernel-coverage claim. Physical pool bindings, L2 placement, and
hardware execution remain separate integration work.

## Zero-copy HMX binding adapter

`prepared_native.c/.h` bridges HMX schedule entries to
`native_runtime/qd_native_direct.h`. It registers only QDR HMX variants 32 and
33. The request builder reads input, weight, Fi, output, and block spans
through bound tensor views, and resolves W/Fi from their current direct pool 1
offsets. It never follows a static-source alias or reads compressed NPRG or
QDRX data during dispatch. It checks storage kind, class, data type, complete
W/Fi shape and padding, and Fi's `[1, CO/32, 1, 64]` shape. The one weight
shape exception is the proved CI32/CO32 profile with logical W/Y output width
2 and physical width 32. Variant 32 is Xpb 1x1; variant 33 is Xwb 1x1 or the
proved Xwb 3x3 shape family. The direct
binder then validates page ranges, layout, and aliases against active VTCM.

The preflight callback calls the direct binder's read-only validation entry
point. The execute callback reruns the binder, which writes the L2 table/ABI
and reserved-tail records, then calls the supplied backend `submit_and_drain`
callback. That callback must enforce the VTCM lease, HMX ownership, completion,
and cache rules. Before issuing the leaf, publish metadata with cache clean for
the full 2,048-byte table region, 120-byte ABI region, and 8 KiB tail header/
guard region, then execute the platform memory barrier. Preserve and restore
SSR bit 26 for each call as required by the proven 3x3 issue path; drain HMX
before returning. After completion, check the adapter's guards and direct
binding invariants before allowing the next command. The host test's callback only checks the binding and counts
accepted commands; it does not invoke HMX or claim numerical equivalence.

For both actual model schedules, metadata validation accepted 132 1x1 and 583
3x3 command records and rejected 131 variant-32/33 records. The test checks
that full-model preflight fails before writes due to missing non-HMX handlers,
read-only HMX validation leaves tables/ABI/tail unchanged, and changing an
accepted command's input max-width makes the same opcode/variant reject.
These are metadata/binding coverage counts, not hardware or numerical kernel
counts.

## InputSlicePad and NOP

`prepared_io.c/.h` registers QDR `NOP` variants 2/6/7/10 and InputSlicePad
variant 2. NOP 2 must carry the duplicate-pad flag and its original seven
input/one output references; NOP 6/7/10 must have no refs and no I/O index.
InputSlicePad requires its seven source refs in the observed shape-value/scalar
forms, a u8 `FB` direct pool-1 output with `d=m=output_shape`, no tensor
padding, a caller-provided input buffer selected by `io_index`, exact source
and destination byte counts, and an output span within the caller's writable
pool-1 ceiling. Preflight computes bounds and checks source/destination
overlap without writing; execution calls the audited direct copy helper. There
is no dense scratch or fallback path.

The host test executes the first contiguous InputSlicePad range from each
actual prepared model (22 primary and 17 alternate commands), compares the
result against the CPU reference loop, and compares all 8 MiB of pool 1 plus
every other pool to prove writes stayed inside the declared output spans. It
also mutates an output offset into the reserved tail and aliases a caller input
with pool 1; both cases reject before changing any pool. Actual duplicate-pad
and framework NOP records also preserve the full pool byte-for-byte.

## InputSlice and Im2col

`prepared_preprocess.c/.h` exposes
`nprg2_executor_register_preprocess`; the runner must register it in addition
to the core-safe I/O handlers. It covers InputSlice variant 1 only for the
observed `*InputSlice@Ff.s4*6.` tuple: six exact shape-value inputs, caller
input 3 containing 128 bytes, and direct float32 pool-1 output `[1,2,4,4]`.
Preflight checks the active pool-1 limit, rejects input/output overlap, and
rejects any of the 32 input words with an all-ones exponent (NaN or infinity).
It copies the accepted words directly.

Im2col coverage is variant 24 (`ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t`)
for u8 source `[1,10,258,1]`, kernel `[3,3,1,32]`, stride `[1,1,1,1]`, and CB
output `[1,8,256,9]` with max shape `[1,8,256,32]`. The output page slice must
contain 32 in-range, unique pool pages inside the active writable extent and
disjoint from the input. Preflight proves these invariants without output
writes; execution calls the existing audited direct im2col helper. It uses no
heap or scratch and rejects other tuples before graph-pool writes.

`run_prepared_preprocess_sim.sh` captures the real CPU prefix before command
0 InputSlice and the first CB Im2col command in both models, runs the prepared
commands on V75, then compares all physical pool bytes to the CPU step.
Both InputSlice commands and both Im2col slices pass this exact comparison;
the alternate Im2col output has slice origin 35, whose active 32-page range
is unique even though other entries in the full table alias. InputSlice
overlap and Im2col active-tail corruption are rejected before writes.

## Bounded prepared runner and L2 budget

`prepared_runner.c/.h` owns session setup/teardown but not physical mapping. The
caller supplies the verified pool pointers, prepared raw image and trusted
hashes, an expanded-block L2 arena, and a hot-table L2 arena. Setup validates
the image and pools, copies the expanded block table once into L2, temporarily
redirects `program.expanded_blocks` to that copy, and binds hot metadata with
`nprg2_bind_metadata`. Close restores the original setup allocation before
`nprg2_close`; it never passes an L2 pointer to the allocator. Setup performs
no graph-pool writes. Execute submits exactly the requested interval to the
registered handler table; the executor validates the entire interval before
running its first command.

The tested model sizes fit the supplied placement plan:

| Region | Planned interval | Primary used | Alternate used |
| --- | --- | ---: | ---: |
| Native tables | `0x30000000–0x30000800` | 2,048 B | 2,048 B |
| Native ABI | `0x30000800–0x30000878` | 120 B | 120 B |
| Expanded blocks | `0x30002000–0x30080000` | 481,320 B | 482,368 B |
| Worker stacks | `0x30080000–0x300B0000` | 192 KiB reserved | 192 KiB reserved |
| Hot program | `0x300B0000–0x300EF000` | 235,208 B | 234,072 B |
| Service | `0x300F0000–0x30100000` | 64 KiB reserved | 64 KiB reserved |

The host test allocates disjoint buffers of those arena capacities, checks
bound block views point into the copied L2 block table, dispatches a real
InputSlicePad command and a real HMX metadata command through the runner, and
verifies full pool boundaries. The HMX backend is a mock that confirms the
direct binding and does not issue a kernel. It also proves an
unsupported interval after several supported InputSlicePad commands rejects
before graph writes, arena overlap rejects setup, and close leaves caller L2
buffers owned by the caller. These host buffers
validate the runner contract and byte budgets; they do not establish linker
placement or the firmware's ownership of every L2 address. The native runner
has no legacy fallback, so commands without registered handlers reject while
pool 1 remains mapped as VTCM.

The public `nprg2_executor_preflight` and `nprg2_runner_preflight` APIs provide
that same read-only coverage/feature/alias/scratch check separately. The DSP
wrapper can call runner preflight before its pool1 boundary copy, then call
runner execute afterward; execute deliberately rechecks before dispatch.
Preflight stats report validated count, failure command/op/variant, and zero
executed commands. On a preflight failure the execute API reports matching
validation stats and performs no handler call.

## Native convolution family mask

`Nprg2NativeConfig.family_mask` narrows HMX coverage before execution. Zero
allows the currently approved 1x1 and 3x3 families; `NPRG2_NATIVE_ALLOW_1X1`
allows only 1x1, and `NPRG2_NATIVE_ALLOW_3X3` allows only 3x3. The mask is
checked by both the read-only handler validator and its execution callback.
With mask 1, any 3x3 command makes whole-range preflight fail before the
runner moves pool 1 or starts an earlier command.

## Blocked u8 elementwise handlers

`prepared_elementwise.c/.h` registers QDR ADD_U8 variant 13, QDR REQUANT_U8
variant 70 (`linearclip_simple_rq@CB*2.fi`), and LinearClip variant 68
(`linearclip@CH.CB.fi`, blocked CB u8 to CH u16). Other variants remain
uncovered. The requant handler uses the existing blocked u8 add arithmetic
with a zero second multiplier, matching the CPU reference's rounding and
saturation. The handlers read current bound tensor locations; LinearClip
also reads current Fi values. None allocates or creates a dense intermediate. They
validate the exact blocked geometry, command coefficients or Fi shape/range,
pool-1 active write ceiling, touched-page bounds, and output/input page alias
rules before execution.

Alias proof uses caller-owned page-address arrays sized for the actual
maximum 2,560 block entries per tensor. The requirement is
`3 * 2560 * sizeof(uintptr_t) + 7 * sizeof(uint8_t *)` bytes: 61,464 bytes on
64-bit hosts and 30,748 bytes on Hexagon. The trailing seven pointers hold the
pool map needed by the audited HVX helpers. Preflight does no graph-pool
writes. The runner must reserve this scratch for command ranges containing
these handlers. The current executor repeats whole-range preflight at run
entry. A separate plan-cache prototype exists, but is not integrated with the
runner and must not replace whole-range preflight: see the limitations below.

The fixture generator runs NativeGraph's CPU backend to the first command of
each registered family, snapshots every graph pool and the live Fi spans,
executes that exact CPU command, and captures the physical output pages. The
V75 simulator harness reloads those pools, dispatches the same prepared
command, and compares every pool byte with the captured CPU step. It also
mutates a touched output page to alias an input and places an output page in
the reserved VTCM tail; both are rejected before any pool write. Numerical
execution calls the existing audited `blocked_glue/add_u8.c` and
`linearclip_u8_u16.c` implementations. This is a simulator proof, not a
device latency result.

`run_prepared_elementwise_host_sanitized.sh` runs ASan/UBSan read-only
preflight against actual LinearClip and requant metadata from both NPRG images.
It also rejects a malformed requant coefficient without changing any pool.
Its host link stubs cover only the unused numeric apply calls; V75 simulation
above executes the real HVX glue.

## Additional bounded command handlers

The independent `prepared_block_zap`, `prepared_spill_fill`, and
`prepared_flat_vtcm` modules expose explicit registration functions. They are
not automatically added to every runner session. A caller must register each
module before requesting a range that contains its command family.

`@BlockZap` (opcode 44, variant 5) accepts the three observed page-copy
headers and checks every 2 KiB source and destination against the active
pool-1 ceiling before writing. A host ASan/UBSan harness runs all 33 commands
in each model and compares the complete pool-1 image with the CPU page-copy
rule after every command. A strict V75 object build passes.

`@Fill` and `@Spill` (opcode 26, variants 8 and 9) copy ordered spans between
pool 6 and the encoded pool. The handler proves every span before the first
copy; Fill destinations in pool 1 must stay within the active ceiling. The
host harness covers all 62 Fill / 26 Spill commands in primary and 53 Fill /
20 Spill in alternate, checking all pool bytes after each command against an
ordered `memmove` oracle. It rejects a malformed later span without writes.
A strict V75 object build passes; no V75 numerical fixture is claimed.

`flat_to_vtcm@Ff.ff.` (opcode 16, variant 65) supports the observed direct
float32 pool-3 to pool-1 copy. Its test executes all 160 commands in each
model, compares full pool-1 and pool-3 images, and checks a reduced write
ceiling rejects before writes. The V75 simulator build and run pass.

`ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4` (opcode 7, variant 31) wraps the
audited direct blocked u8 layout helper with an 8 KiB caller scratch tile.
Its read-only preflight accepts all 174 commands in each model and restricts
the helper's planned output pages to the active pool-1 ceiling. Host and V75
simulator runs compare the full pool-1 image against captured CPU output for
10 distinct shape/view profiles per model. These are bounded command proofs,
not a full prepared schedule or device timing result.

## Additional bounded CPU-prefix handlers

These modules use explicit registration calls; adding a source file to a
build does not make its opcode/variant available in a runner. Each proof below
covers the named fixture commands or tuple only. It does not imply full model
schedule coverage, Frame integration, or device execution.

`prepared_abs.c/.h` registers QDR ABS variants 12 (`Abs@FH*2`, direct u16)
and 11 (`Abs@CH*2`, blocked Crouton u16). The V75 simulator test compares
physical pools with captured CPU prefixes for primary FH and CH commands and
alternate FH. It checks malformed dtype/shape, unsupported variant, bounds,
flat-buffer aliasing, and blocked-view corruption fail closed. This is a
small set of command fixtures, not coverage of every ABS record.

`prepared_mean.c/.h` registers QDR Mean variant 47 for exactly three direct
u16 inputs and one direct u16 output with 32 input elements, one output
element, and matching quantization metadata. Primary and alternate CPU-prefix
fixtures pass bit-for-bit under the V75 simulator. The test also rejects an
unregistered variant and a destination beyond the writable pool-1 limit.
This does not establish generic Mean coverage.

`prepared_conv_s2.c/.h` registers HMX-convolution variant 36, the tested
Crouton u8 stride-2 profile. Its V75 simulator test compares full physical
pools with captured NativeGraph CPU prefixes for three primary commands
(command indices 224, 363, and 3407) and tests rejection of an unsupported
variant. This is command-level software/HVX equivalence for those fixtures;
no alternate-model or device result is claimed.

`prepared_conv_u16.c/.h` registers HMX-convolution variants 34 and 35 for the
tested CH/u16 1x1 and 3x3 forms. The simulator harness captures two primary
model CPU-prefix fixtures and compares the resulting physical pool images.
It is not a claim of full u16 convolution family coverage or device HMX
execution.

`prepared_slice_pad.c/.h` is an independent InputSlicePad registration module
for opcode 2 variants 2 and 8. The Oct 31 report and simulator fixture test
cover only variant 2 at alternate commands 68 and 69 (sequences 118 and 120),
with exact CPU-prefix output and fail-closed checks. Variant 8 is registered
in code but has no numerical fixture evidence in that report. This module is
separate from the existing `prepared_io` InputSlicePad variant-2 handler;
callers should register only one owner for a given key in a registry.

## New direct scalar handlers

All four modules below register explicitly and are not wired into the Frame
runtime. They do not establish a multi-command or device result. Each host
test first runs preflight on every matching command in both models. It then
loads captured CPU-prefix pools, runs one command, and compares all seven
physical pools byte-for-byte. Malformed metadata must reject before any pool
write. The V75 simulator runners rebuild the same tests for `-mv75`.

`prepared_sign.c/.h` registers opcode 30, variants 40 (`ElementWiseSign@FH*2`,
direct pool-1 FH/u16 scalar) and 39 (blocked CH). Census: 13/13 per model.
Fixtures exist for all 13 commands per model. The default build compares
primary 157/4776 and alternate 109/4666. Building with `-DSIGN_ALL_FIXTURES`
compares all 26 (host: all pass). V75: the four default cases pass. The CH
path validates page bounds, output uniqueness, and read/write disjointness
using two 2,560-entry `uintptr_t` stack arrays (20 KiB on Hexagon) and an
O(n^2) overlap scan, and `run` repeats that validation. Check this against the
per-worker stack budget before runtime integration.

`prepared_addsub_flat.c/.h` registers opcode 42, variant 62 for the three
direct flat Add/Sub commands in each model: primary 152/180/182, alternate
100/128/130. The arithmetic profiles are one subtraction and two additions,
mirrored across models. It validates offsets, shapes, quantization, the call
signature, coefficients, pool roles and bounds, output aliasing, and the write
ceiling. Host: 6/6 full-pool exact. V75: 6/6 full-pool exact.

`prepared_mul_flat.c/.h` registers opcode 41, variant 75
(`mul_op_flat@FH*2.fH.fi`) for the two direct scalar sites per model: primary
158/181, alternate 110/129. It reads the live six-word parameter tensor and
checks geometry, dtype, bounds, aliasing, and the pool-1 ceiling. Host and
V75: 4/4 full-pool exact. CH Mul (10 per model) is not covered.

`prepared_reshape_cb.c/.h` registers opcode 16, variant 49 (`Reshape.hw@CB*2`):
source `[1,1,64,128]` (max H 8) to output `[1,8,8,128]`. The single live
source row is the source H padding (0–7). Each value occurs in 10 commands per
model. The handler copies 32-byte runs from 32 source pages to 4 output pages.
It requires pool-1 pages that are in bounds, unique, and disjoint from the
source. Census: 80/80 per model. `capture_reshape_cb_cpu_prefix.py --all-pads`
captures the first command for each padding value. Host compares all 16
(`*-reshape-cb-pad{0..7}.epcpu`). V75 (`-DRESHAPE_SIM_SUBSET`) compares pads
0, 3, and 7; all pass. Caveat: in the smoke input, source rows 0–5 hold
identical data. A mutant that ignores the padding and reads row 0 still
passes pads 1–5 and first fails at pad 6. So only pads 6–7 actually test row
selection.

Reproduce with `sh frame/native_program/run_prepared_{sign,addsub_flat,mul_flat,reshape_cb}_sim.sh`
from the project root. For host runs, compile `test_prepared_<family>.c` with
its handler, `prepared_executor.c`, and `nprg2_runtime.c`.

## Full-model reference kernel

`prepared_reference.c/.h` is an exact port of every `qd_graph.c` CPU kernel.
It covers all 76 (opcode, variant) keys, including the NOP duplicate-pad key
(op 1, variant 2), so both prepared models run every command through the
executor. Each command first runs a dry pass (metadata, call arguments, every
touched block/byte range against pool bounds and the pool-1 ceiling, and the
exact scratch need), then a wet pass. Gather axes, Transpose permutations and
DepthToSpace factors stored as direct pool-5 constants are read in the dry
pass, so their results are sized exactly. Other data-dependent checks
(indices, runtime shifts) run in the wet pass before that command's first
graph write. Convolution is a slow scalar CPU port: a correctness path, not a
performance path.

Registration is explicit. `nprg2_executor_register_reference_missing` fills
every free key. `nprg2_executor_chain_reference` puts the reference kernel
under an existing specialized handler, for commands that handler's validator
rejects. The chain storage must outlive the executor. Scratch must come from
`nprg2_exec_reference_scratch` over the scheduled range. The peak is 1,647,936
bytes in both models (op 6 variant 33), well above the 0x8000-byte L2 scratch
`prepared_run` passes today.

Fast exact convolution: `nprg2_executor_register_reference_conv` registers
every op-6 variant with an `Nprg2ReferenceConv` workspace (at least 467,456
bytes; tests use 0x7e000). Each convolution then reads raw weight tiles in
place and accumulates in int32 (HVX `vrmpy` when built with `-mhvx`, portable
C otherwise), with the unchanged `requant`. The decision depends only on the
shapes, and anything the fast path does not support falls back to the scalar
reference. Size scratch and the workspace with
`nprg2_exec_reference_conv_measure`; the scratch peak drops to 983,168 bytes.
Pools that overlap the workspace are clipped below it in the dry pass.

Evidence:

- Host, `test_prepared_full_model.c`, both models: reference-only and mixed
  (18 specialized keys chained over the reference) both pass. Lockstep shows
  all seven pools identical to `qd_graph` CPU after every command. A whole
  schedule runs as one preflight with no pool writes, then one run. Both
  produce the emulator smoke output sha256. In reference-only lockstep, each
  command's scratch is capped at exactly its own measured need. No command
  writes above the 0x780000 pool-1 ceiling.
- V75 simulator, `run_prepared_reference_sim.sh` (`test_prepared_reference_sim.c`):
  `prefix` mode runs primary commands 0–131 and alternate commands 0–99 from
  the `.mpfx` CPU captures. All seven pools match bit for bit in both models
  (about 4 minutes including the build). `full` mode runs the whole schedule
  from zeroed pools plus constants and checks the output sha256. With `fast`
  and `segments 6`, all 12 segments of both models pass, and the final outputs
  match the emulator sha256. That run took about 6.5e9 handler pcycles per
  model, 73% of it conv.
- Host: `fast` and `mixedfast` modes of `test_prepared_full_model` pass
  lockstep and whole-schedule for both models.

Vector block copies and direct blocked conv output (2026-11-01, **not yet
run in the simulator**; HVX, `-DNPRG2_SCALAR_REQUANT` and host builds compile
with `-Werror`):

- In the u8 blocked layout, pixels W..W+7 (W%8==0) of one row and one
  32-channel group are 256 contiguous bytes, 32 per pixel. `blocked_impl`
  (behind `xfer_copy`, so conv input staging and every reference op) now
  moves such 8-pixel runs with `blocked_run8`. With 32 dense channels it does
  two 128-byte vector copies. With more channels it does one 32-byte masked
  store per pixel (`store_part`/`load_part`). Partial channel groups and
  absent blocks keep the scalar path, which zero-fills on load and skips on
  store. `load_part` loads only the aligned vectors holding the first and
  last byte, so it never reads into an unmapped next page. u16/i32 blocked
  copies are unchanged.
- `near_compute`, u8 with fused HVX requant, blocked output, and a chunk
  starting at an 8-aligned output column: `rq_emit` stores each block's
  pixels straight into the output block (`RqOut.blocked`/`blk[2]`,
  located by `blocked_row8`). Full 8-pixel blocks are two aligned vector
  stores. The near output chunk and its `xfer_copy` are skipped, so
  phase 3 (output write) drops to about zero and the store cost shows up in
  phase 1. u16, scalar requant and unaligned chunks still use the near chunk
  and `xfer_copy`.
- To validate: `run_prepared_reference_sim.sh` in prefix mode, then fast full
  with `segments 6`, comparing pools and sha256 against the earlier results.

Fused HVX requant in registers, single-multiply u8 (2026-11-01, **simulator
unit tests and host only; not run in the full-model simulator or on
hardware**):

- `RQ_VECS` is 8. Per 32-channel block, `rq_prepare` stores u8 {off, pre, M,
  post, r, c, 0, 0} and u16 {offh, off&255, M, post, r, c, sl, 0}, where
  r=2^(post-1), u8 c=2^max(post-2,0), u16 c=min(2^(post+6),2^22) and
  sl=16-post. `rq_emit` no longer recomputes them.
- u8, every admitted post (1..22): clamp x to [0,c], then
  q=vlsr(vmpyieacc(r,x,M),post). c*M>=2^(post+8), so q(c)>=256 saturates.
  x*M+r<=2^(post-2)*2047+2^(post-1)<2^32, so the product taken mod 2^32
  with a logical shift is exact. There is no per-command selection, and the
  16-bit split path is now u16 only (sr dropped, since post<=16).
- `rq_lane`/`rq_emit_k`/`rq_emit` are always_inline. nb and sixteen are
  compile-time constants in every copy (so no `__hexagon_udivsi3`), and every
  loop has a constant trip count with np only in guards, fully unrolled.
  Each output vector's lanes are formed as it is packed. Full 8-pixel blocks
  inline into `tile_block`. Partial blocks pass their sums through `acc`
  (128-byte aligned, as `lo` already was) to the out-of-line `rq_emit_part`.
  V75 -O2: `tile_block` frame 2304 -> 16 bytes, stack vector accesses
  14 -> 0, and the `rq_emit` call (2688-byte frame, 27 stack vector accesses,
  divide calls) is gone. Code grows about 19 KB (`tile_block` 1.2 -> 10 KB,
  `rq_emit_part` 12 KB).
- Parameter memory doubles (co*32 bytes). On the host, both models still put
  all 1036 convs on near memory with a 169,216-byte workspace peak. Only the
  80 u8 s2 convs with ci=128, co=64, ow=128 change near chunk width
  (80 -> 72 columns, still 2 chunks per row).
- Evidence: an exhaustive lane check (`rq32b.c` style: all M 1024..2047, u8
  post 1..22, u16 post 1..16, dense sweeps at 0, the clip edge, both clamp
  edges and a coarse int32 sweep; 3.5e9 cases) gives 0 mismatches. V75
  simulator `.work/rqtest/test_rq.c` (800 random commands, now including
  blocked output with absent blocks and misaligned destinations, and u8 post
  19..22 with edge M across the clamp and clip edges) passes 240,512 outputs.
  Mutants with a too-small clamp or a wrong blocked offset fail it. A
  mutant that ignores a NULL (absent) `blk` still passes: only the
  simulator's NULL-dereference warning shows it, so absent-block skipping is
  weakly covered. Host
  `test_prepared_full_model` fast/mixedfast and `test_refsim both full fast`
  pass for both models. The engine builds with `-Werror`, with and without
  `-DNPRG2_SCALAR_REQUANT`.
- `.work/rqtest/bench_rq.c`, pcycles per 8-pixel x 64-channel block (fused
  total, requant part), before -> after: u8 1x1 2200 -> 775 (1659 -> 225);
  u8 3x3 ci=64 9136 -> 7711; u16 1x1 2983 -> 1630 (1914 -> 543); u16 3x3
  9511 -> 8158. That is about 14 cycles per output vector for u8 and 34 for
  u16.

HVX elementwise handlers back on in the reference build, via per-handler
scratch regions (2026-11-01, **change made, NOT run in the simulator or on
hardware**):

- Why: about 27% of simulated time is the generic scalar reference over
  ~138M elements per model; the largest are ADD_U8/13 (15.6M elements),
  S2D/31, REQUANT_U8/70 and LINEARCLIP/68. `prepared_elementwise.c` already
  has HVX handlers for 13, 70 and 68 from the accepted 74d engine. The
  reference build turned them off (`#error` in `qd_graph_prepared.h`, check in
  `build_resident_dsp.sh`) because they and the conv near state both used the
  L2 slot at 0x30078000, and the reference build hands every handler its
  L1-cached heap scratch.
- Executor (`prepared_executor.{c,h}`): `NPRG2_EXEC_SCRATCH_REGIONS` (2).
  Region 0 is the request's `scratch`/`scratch_bytes`; region 1
  (`NPRG2_EXEC_SCRATCH_NEAR`) is `request->extra_scratch[0]`. Handlers pick a
  region with `nprg2_executor_register_scratch_region` (stored in the former
  `Nprg2ExecHandler.reserved` byte, now `scratch_region`; the old register
  calls mean region 0). Preflight checks the handler's byte demand against
  its region and run passes that region as `call->scratch`. Every region is
  checked against graph memory, the control arrays, I/O buffers and the
  other regions. `Nprg2ExecCall.scratch_region[]` carries all regions.
- Reference chain (`prepared_reference.{c,h}`): `Nprg2ReferenceChain` now
  stores the wrapped handler's region and byte demand. The chained entry is
  moved to region 0 with zero demand (the reference validator checks its own
  peak). `chain_call` gives the wrapped handler its own region. If that region
  is absent or too small, or the handler's validator rejects, the command runs
  the reference kernel with region 0, as before.
- Runner: `Nprg2PreparedRunner.extra_scratch` and
  `nprg2_runner_set_scratch_region`, copied into every request.
- Plan cache: builds region 0 into its calls and rejects
  (`NPRG2_PLAN_UNSUPPORTED`) a handler registered for any other region.
- Elementwise: `nprg2_executor_register_elementwise_region(e, region)`;
  `nprg2_executor_register_elementwise` is region 0, as before.
- Frame (`qd_graph_prepared.h`): with `QDS_PREPARED_REFERENCE=1` and
  `QDS_PREPARED_ELEMENTWISE=1`, the handlers are registered for region 1 =
  `QDS_ELEMENTWISE_L2` 0x300f8000..0x30100000 (32 KiB L2-TCM, demand 30,748
  bytes). That is the top half of the work slot: `QdsWork` uses 116 bytes at
  0x300f0000, the hot program ends at 0x300ef000 and the worker stacks at
  0x300b0000. They stay chained, so region 0 is still the 1 MiB heap scratch
  and conv near stays at 0x30078000. The `#error` is gone, and
  `build_resident_dsp.sh` no longer rejects REFERENCE=1 with ELEMENTWISE=1
  (VECTORIZE=0 and COMPACT_3X3=0 are still required). The non-reference
  elementwise build is unchanged (region 0 = 0x30078000).
- Checked: host ASan/UBSan `test_prepared_elementwise_host` (now also tests
  region 1: absent or too small -> REJECTED, aliasing region 0 -> ARGUMENT,
  sized -> OK), `test_prepared_executor` and `test_nprg2_plan_cache` on both
  models all pass. The Hexagon engine with the 74d flags minus COMPACT_3X3
  plus VECTORIZE=0, REFERENCE=1 and ELEMENTWISE=1 compiles and links. It was
  built from a scratch copy of `frame/`, so `frame/build` and the bootstrap
  were not touched.
- Not done: the reference-simulator full run with 13/70/68 on HVX, and any
  hardware run. Before relying on it, recheck 13/70/68 against the CPU
  reference on hardware (pools and sha256 as for the frozen reference
  engine), and confirm that nothing on the Frame writes 0x300f8000..0x30100000
  during a prepared call.

Direct data-movement fast paths, HVX weight copy, near scratch (2026-11-01,
**host lockstep and V75 simulator unit tests only; the full-model simulator
gate is recorded below; not run on hardware**):

- `prepared_datamove.inc` (included by `prepared_reference.c`) adds exact
  block-to-block / row-to-row paths that skip the dense int64-agnostic temps
  (`t_read` -> permute -> `t_write`). The choice uses only geometry and
  immutable pool-5 constants, so the dry and wet passes agree and a taken
  path needs no call scratch. Each runs the same `xfer_check` bounds checks
  first, checks all indices before its first graph write, and falls back to
  the generic kernel for anything uncovered or when source and destination
  blocks/byte ranges overlap (the generic path reads everything before it
  writes). Vector moves use `DV` (HVX on Hexagon, a 128-byte array on the
  host, so lockstep runs the same index logic). Vector paths need 128-byte
  aligned pool bases. `-DDM_ENABLED=0` compiles them out (baseline).
  - SpaceToDepth (variant 31): u8 blocked in and out, ci a multiple of 32,
    oc=4*ci, pw%8==7, ow%4==0. A rolling window of three input vectors per
    4 output pixels, `valign` + 32-byte `vdeal`, full aligned vector stores.
  - ResizeNear (51): u8 blocked, column map w>>1 (parameters {0x80000000, v4 <
    0x80000000, 0}, read from the constant), any row map, ow%8==0, C%32==0;
    one load becomes two stores via 32-byte `vshuff`.
  - Transpose (55-58): blocked u16 H<->W (perm 0213, even sizes) via
    `vshuffe/vshuffo` halfword pairs; flat (direct) u8/u16/i32 any
    permutation, 16x16 tiles. Blocked W<->C swaps (55 u8, 56 u16 0132) and
    channel counts below 32 stay generic.
  - Slice (54), SlicePad flat (53), Gather flat (46): direct pool-to-pool row
    copies with contiguous-run merging (`qd_vcopy` for runs >= 512 bytes).
    Blocked SlicePad (52, channel shift inside a 32-byte pixel) stays generic.
- `prepared_vcopy.h`: `qd_vcopy`, a memmove-compatible HVX copy: eight vector
  loads in flight, aligned destination stores, `valign` for a misaligned
  source, scalar head, masked tail; a destination inside the source range
  uses memmove. Used by `nprg2_exec_weights_run` (the built-in
  WEIGHTS_TO_VTCM handler), `k_weights_to_vtcm`, and the flat copies.
  `-DQD_NO_VCOPY` restores memmove.
- Near scratch: `nprg2_exec_reference_run` already runs a dry pass, so it
  knows the exact scratch need; if it is <= scratch region 1
  (`NPRG2_EXEC_SCRATCH_NEAR`) the wet pass uses that region instead of the
  DDR heap. On the Frame, region 1 is the existing `QDS_ELEMENTWISE_L2`
  0x300f8000..0x30100000 (32 KiB, built with REFERENCE+ELEMENTWISE). It is
  free during a reference command because commands run one at a time and the
  elementwise handlers use it only inside their own commands. No new L2
  range was added; `qd_graph_prepared.h` is unchanged.
- Tests: `test_prepared_datamove_sim.c` (`run_prepared_datamove_sim.sh
  [cycles]`; `SDK_MOUNT=<.work/hexagon-sdk path>` for a worktree) runs random
  cases (absent blocks, slice offsets, shuffled placement, misaligned pool
  bases, aliasing, declining shapes) against the generic kernels and compares
  the whole pool, result codes and scratch need; also vcopy at all
  alignments/overlaps. `test_prepared_full_model` takes `QD_NEAR=1` to supply
  the near region. `run_prepared_reference_sim.sh` accepts `SDK_MOUNT` and
  `CFLAGS_EXTRA`.

HVX u16 elementwise handlers: REQUANT_U16/71, ADD1111/59, MUL_U16/74, ADD_U16/14,
ADDSUB_U16/60, MUL1111/73 (2026-11-01, **simulator and host only, not on hardware**):

- Files: `blocked_glue/elementwise_u16.{h,c}` (derivation, validator-side
  parameter checks, HVX kernel), handlers in `prepared_elementwise.c`
  (`w16_*`, registered with the others, same region and the same 30,748-byte
  demand, so the 32 KiB region-1 fit is unchanged), `elementwise_host_glue_stubs.c`,
  `build_resident_dsp.sh` (glue source list), `test_ew16_lanes_host.c`,
  `blocked_glue/elementwise_u16_sim.c`, `run_prepared_reference_sim.sh`
  (compiles the handlers), `test_prepared_reference_sim.c` (new `ew` option
  registers and chains them; new host mode `tail <first> <dir>` writes one
  segment covering the late schedule).
- Elements taking the reference path per model (output dims product, after
  the existing 13/70/68 and specialised handlers; both models have identical
  op/variant tables apart from 71): S2D 7/31 10.0M, RESIZE 24/51 5.9M,
  TRANSPOSE 18/56 2.9M, IM2COL 5/24 1.47M, REQUANT_U16 9/71 1.15M (alt 1.31M),
  TRANSPOSE 18/55 0.98M, ADD1111 13/59 0.98M, MUL_U16 14/74 0.82M, 19/15 0.70M,
  35/23 0.66M, 16/44 0.66M, 16/41 0.57M, 18/57 0.51M, SIGN 30/39 0.49M,
  LINEARCLIP 39/67 0.49M, ADDSUB_U16 12/60 0.49M, ADD_U16 11/14 0.33M,
  5/25 0.33M, 16/50 0.33M, MUL_INLINE 41/72 0.16M, MUL1111 15/73 0.16M; the
  rest are smaller. Whole non-conv total about 61.6M per model, 20M of it
  already HVX (10/13 7.8M, 8/70 9.6M, 39/68 2.6M). The doc's earlier "138M" is
  both models together. Data movement (S2D, resize, transpose, im2col) is the
  larger remaining share but is not elementwise and was not touched. Not
  done: 41/72 (parameters in a pool tensor, scalar stored as a crouton),
  39/67 and 39/69 (linearclip u16->u16, FB), SIGN 30/39.
- Schedule geometry: every one of these commands is a full CH tile
  1x8x{64,80}x{32,64}, origin 0, d == m, nothing absent in the output.
  Observed parameters: REQUANT_U16 (m,s,zin,zout) = (16384,14,0,0),
  (16384,14,32768,32768), (16548,14,0,0), (32765,15,0,0); ADD1111
  (k0,k1,r6,r7) = (16384,16384,13,0xE0000000), (16384,8192,13,0xF0000000);
  MUL (P1,P2) = (262127,262119), (262148,262140), (262156,262148),
  (262281,262273), (284378,284369), K=0x80010000, zb=za=0; ADD (k0,k1,r6) =
  (13699,26008,14), (16995,14224,13); ADDSUB (k0,k1,r6,zp) =
  (14379,18491,14,36775), (17144,3455,13,10701), (30497,30497,14,61378).
- All six ops are lane independent, so the CH interleave is irrelevant: a
  128-byte half block is 64 u16 lanes, stored under a pixel/channel mask
  (same masks as the u8->u16 LinearClip glue). The validator requires equal
  dims, origin[2]%4 and origin[3]%32 equal across tensors, the unpadded depth
  form of Memory.write, disjoint output/input blocks, and rejects everything
  else (so it falls back to the reference kernel).
- Exactness (32-bit lane forms of the int64 formulas; `make_*` return 0
  unless no intermediate can overflow for any u16 input):
  requant: p=(v-zin)*m+2^(s-1) checked in int32 at both extremes of v (m>=0 so
  monotone), q=(p>>s)+zout checked in int32, saturating pack.
  ADD1111: (x*k0+c)>>k2 with c=s*k1+k3 (runtime scalar), all four (x,s) corners
  and c checked in int32. ADD: unsigned sum < 2^32 checked, logical shift
  (k2+1<=31), so result < 2^31. ADDSUB: sum range [k3-65535*k1, k3+65535*k0]
  fits int32 (arithmetic shift) or lies in [0,2^32) (logical shift, kind
  ADDSUBU; (30497,30497,14,61378) needs this). MUL: D=a*b+K mod 2^32; with
  D=65536*Dh+Dl, e=Dh*lo+((Dl*lo)>>16); q=2*hi*Dh+(e>>15)+(((e&0x7fff)+hi*Dl)>>15)
  (no clip needed for |hi|<=16384); s=sat32(q+P2); out=pack_sat(((s>>2)+1)>>1)
  since (s+4)>>3=floor((floor(s/4)+1)/2). Observed lo reaches 65519, so the
  naive R=2*hi*Dl+2*Dh*lo overflows; the e>>15 split avoids it. zb/za must be 0.
  `test_ew16_lanes_host` (portable C, lane forms copied op for op): every
  observed parameter set over the complete input space (requant all 65536 v;
  ADD1111/ADD/ADDSUB all 2^32 (a,b) pairs; MUL all 2^32 (a,b) and, separately,
  all 2^32 values of D, including four P2 values at the int32 limits so the
  saturating add engages), plus 92k random requant, 78k ADD, 45k ADDSUB (2.8k
  unsigned) and 228 MUL parameter sets accepted by `make_*` (all inputs or
  dense samples). 0 mismatches (138 s CPU, 15 s on 14 cores).
- Simulator: `elementwise_u16_sim` checks the HVX against an independent
  int64 oracle that also does the CH addressing element by element, with random
  geometry (per-tensor meta/origin/slice, partial dims and pixel pairs,
  batch 2, absent blocks in inputs and outputs, untouched bytes preserved),
  schedule and random accepted parameters, special-value inputs: 268 trials
  pass. Mutants (wrong shift, logical vs arithmetic, odd-pixel mask) fail it;
  a non-saturating `q+P2` mutant only fails with the P2-near-limit entries.
  Real data: `test_prepared_reference_sim segment` on the late schedule
  (`tail 4500`, commands 4500..end, primary and alternate) with `fast cycles
  ew`: all seven pools and both outputs identical to the host end state, with
  all six ops on the HVX path.
- Cost (V75 simulator pcycles per output element, primary tail segment;
  reference -> HVX): MUL 112.3 -> 9.8, ADD1111 69.3 -> 6.4, REQUANT_U16 59.0
  -> 6.1, ADDSUB 78.7 -> 7.5, ADD_U16 82.2 -> 8.7, MUL1111 96.3 -> 8.0.
  Tail segment handler total 815.2M -> 535.3M pcycles (primary), alternate
  481.6M with HVX; the six ops go from 307M to 28M. Per schedule that is about
  280M pcycles saved of ~6.5e9. Host: `test_prepared_elementwise_host`
  (sanitized) accepts every one of these commands in both schedules,
  `test_prepared_full_model` mixed/mixedfast both models pass, the Frame engine
  with QDS_PREPARED_REFERENCE=1 QDS_PREPARED_ELEMENTWISE=1 builds (scratch copy).

Direct input staging, row ring, wider direct output, compact MAC tiles
(2026-11-01, **host + V75 simulator only (prefix mode `fast`, unit tests); not
run on hardware**):

- Input staging (`stage_row8`): u8 blocked input is staged straight from the
  blocked layout into near, one 8-pixel run (256 contiguous bytes per
  32-channel group) at a time, zero for absent blocks. ci=32: two vector
  copies per run; ci=64: `vshuff` of 32-byte units; ci%128==0: a 4x4
  transpose of 32-byte units (two `vshuff` stages), stored per pixel; other ci
  (12, 20, ...): 32-byte (or partial) `store_part` pieces. For ci%128 != 0 the
  row keeps `lead` = (input column)%8 pixels in front so runs stay 128-byte
  aligned; for ci%128==0 only the asked pixels are written. The workspace copy
  of the input is gone for these commands (a token 128-byte workspace
  allocation remains so the measured workspace stays nonzero).
- That is only safe if output blocks cannot overwrite input a later row still
  needs, so `blocked_overlap` compares the block sets (bitmap over 2048-byte
  blocks of pool 1; anything else, or unaligned, counts as overlap). On overlap
  and for u16 the old path runs: whole-input `xfer_copy` into the workspace,
  then rows are copied/split (`near_copy`, `near_split16`) into near.
  `near_copy`/`near_split16` now write exactly n bytes (they used to write up
  to 127 past the end). On the host every u8 conv in both models is direct;
  the u16 ones use the workspace copy.
- Row ring: each chunk is processed column-chunk outer, output row inner. A
  new input row is staged per output row (kh-1 in front), into slot r%kh, so
  each input row is staged once per chunk, not kh times. The tap list has one
  segment per kernel row and `tile_block` takes one base pointer per segment,
  so there are no per-phase offset tables. Near layout: `wp` pair tables
  replace `tiles`; `xoff` is nt entries, `segend` kh. The tightest conv (3x3
  ci=128 co=128 ow=32..64) now uses exactly 32,768 of the 32,768 near bytes
  (cols 32), 3x3 ci=256 30,080; all 1036 convs still fit near in both models.
- Compact taps: `count_taps`/`build_taps` drop the stride-2 NO_TILE taps, so
  `tile_block` has no per-tile check. Each tap carries both blocks' weight
  addresses (`wp[2*i+b]`, absolute on 32-bit targets, offsets on the 64-bit
  host); `WREF`/`WPTR`. Measured: an offset+add per operand costs as much as
  the NO_TILE check did, absolute addresses in a pair table do not.
  `tap_wsum` is HVX (vrmpy with ones) instead of 128 scalar adds per tap.
- Direct blocked output now covers u16 (pixel pairs via `vsat` interleave,
  one 128-byte store per pair) and any first-column alignment: `RqOut.a` is the
  first pixel's slot, `blk[b][i]` the i-th block touched, partial stores are
  masked `store_part`s; NULL (absent) blocks are skipped. Aligned full blocks
  keep the old inline path (`aligned` is a compile-time constant); misaligned
  ones go through the noinline `rq_emit_part`. u16 with an odd first slot or an
  odd chunk width still uses the near chunk + `xfer_copy`. The accumulators
  are written to `acc` before calling `rq_emit_part`: passing the local array
  made V75 -O2 spill the tile again (frame 16 -> 2304 bytes).
- Shared helpers changed: `near_copy`, `near_split16` (exact tails), `NearPlan`
  and `tile_block`'s signature (conv path only). `blocked_impl`, `xfer_copy`,
  `store_part`, `load_part` are unchanged.
- Rejected by measurement (bench, 3x3 ci=64, cycles per 8x64 block): scalar
  operand `vrmpy` (no vsplat; weights xor 0x80, needs the 128*sum(a)
  correction) 4432 vs 5710 without correction but 5758 with it; sliding splat
  window for kw=3 5149 (compiler does not pipeline it); one-tile lookahead and
  unrolling made it worse. The simulator charges about 3 cycles per packet of
  two vrmpy and about 1 per vsplat, so splats are what is left.
- Tests: `.work/rqtest/test_rq.c` (tile_block with segments and pair tables vs
  scalar sums + `requant_value`; blocked output u8/u16, every first slot,
  absent blocks, canary pool: every unwritten byte must be intact;
  `stage_row8` for ci 12/20/32/64/96/128/256 with random absent blocks and all
  alignments; `near_copy`/`near_split16` exact length). A mutant without the
  NULL guard makes the simulator report NULL dereferences (run.sh turns that
  into FAIL; it also times out). `bench_rq.c`, pcycles per 8-pixel x 64-channel
  block, before -> after (mac / fused): u8 1x1 550 -> 484 / 775 -> 706; u8 3x3
  ci=64 7486 -> 5836 / 7711 -> 6058; u16 1x1 1087 -> 955 / 1630 -> 1495; u16 3x3
  7615 -> 6043 / 8158 -> 6583.
- Prefix simulator (`run_prepared_reference_sim.sh both prefix 6 fast cycles`,
  set `DOCKER_EXTRA` for an extra docker mount; SDK mount in a worktree): primary 0..131: conv handler
  9.55M -> 7.37M pcycles, staging 4.88M -> 2.37M, mac 3.78M -> 3.81M (flat: the
  prefix convs are small-tap, overhead-bound), output 21K; alternate 0..99: conv
  5.17M -> 3.83M, staging 2.54M -> 1.23M, mac 1.96M -> 1.98M. All seven pools
  match in both models. The remaining ~1.2M per prefix is setup outside the
  phase counters (taps, requant parameters, dry-pass checks).
- Hardware should check: staging and MAC phases against the simulator ratios;
  that no command takes the workspace path where it did not before
  (`conv_path`); VTCM stores of the direct staging (aligned 128-byte vector
  stores into L2-TCM, loads from VTCM at 128-byte alignment).

Not established: anything on hardware. The Frame wiring (`QDS_PREPARED_REFERENCE=1`
in `qd_graph_prepared.h`) is compiled offline only. The pure scalar reference conv
has not completed a full-schedule simulator run.

## Plan-cache prototype

`nprg2_plan_cache.c/.h` is a standalone prototype, not part of
`prepared_runner` and not a supported optimization or replacement for normal
preflight. Its test builds a cache for both actual schedules using mock
handlers that accept every encountered opcode/variant, then checks cached and
ordinary dispatch counts and a couple of mutated-request failures. That
proves the prototype's basic tag-table plumbing only; it does not validate the
real handlers' shape, alias, or live I/O predicates at cached run time.
`dynamic_command_count` is currently left zero by the builder, so cached
preflight skips handler validation. The header's approximate timing comment
is not measured evidence. Use the ordinary executor preflight/run path for
prepared execution.

## Per-op hardware profile and executor overhead (2026-11-01)

Why `variant_profile` / `top_variants_ms` were empty on Frame: the profile
clock and arrays lived in `Nprg2ExecStats`, and `nprg2_executor_preflight`
does `memset(stats,0)`. `prepared_run` set them before preflight, so run saw a
NULL clock and timed nothing. (Variant-only keying also hid opcodes.) Python
did read the right arena slot.

Changes:

- `Nprg2ExecProfile` (new, `Nprg2ExecRequest.profile`, set on the runner with
  `nprg2_runner_set_profile`): clock plus caller-owned arrays, never cleared by
  preflight. Per registry slot (so per opcode+variant): run cycles, run count,
  validator cycles; totals for preflight, validators, dispatch (run loop outside
  handler run: decode, lookup, cancel, scratch) and handler run. Zero cost when
  no clock. On Frame (`qd_graph_prepared.h`, build flag `QDS_PREPARED_PROFILE`,
  default 1, env var passed by `build_resident_dsp.sh`): about two pcycle reads
  per command in preflight and in run, ~1.2M simulator pcycles per schedule
  (~0.7 ms of ~3.5 s). `QDS_PREPARED_PROFILE=0` removes it.
- Export: `prepared_export_profile` writes 64 header bytes plus 128 x 32-byte
  slot records to arena offset `QDS_PROFILE2`=20480 (layout in the function's
  comment). `resident_dsp.py` exposes `dsp.handler_profile` (opcode, variant,
  count, run_ms, validate_ms, no_effect; sorted) and `dsp.executor_profile`
  (preflight_ms, validate_ms, dispatch_ms, run_ms, preflight_calls,
  preflights_reused, executed/validated commands). `verify_prepared_full_model.py`
  stores both in each run row and prints `top_handlers_ms`
  (`"op/var": [count, run_ms, validate_ms]`) and `executor_profile_ms`.
  The old by-variant `variant_profile` now fills too.
- Overhead reduction: (1) `Nprg2ExecRequest.reuse_preflight` (runner:
  `nprg2_runner_set_profile(...,reuse=1)`): execute skips its repeated
  preflight when the same stats object holds the signature (hash of program,
  bound, pools, ceiling, I/O and scratch buffers, interval, registry) of a
  successful preflight of the identical request; any difference runs the full
  preflight. (2) Run calls set `Nprg2ExecCall.preflighted`; the reference
  handler skips its repeated dry validation pass in run. (3) `no_effect`
  handlers (the NOP variants 2/6/7/10, `nprg2_executor_mark_no_effect`) are
  counted but not called in run. (4) Command decoded in place (no memset or
  struct copy per command). Block-table lookups are already O(1)
  (`nprg2_bound_tensor_at` is pointer arithmetic), so no cache was added.
  Preflight still validates every command before the first write.

Measurements (V75 simulator pcycles, `bench_prepared_overhead_sim.sh`, both
models equal within 1%; primary shown, 5068 commands, 1299 NOPs):

| | before | after |
|---|---|---|
| stub preflight | 1,154,697 | 1,161,552 |
| stub run = preflight + dispatch | 2,397,519 | 2,373,999 (dispatch 1.21M) |
| stub run with preflight reused | n/a | 1,222,380 |
| tensor-decoding stubs, preflight | 4,040,748 | 4,047,603 |
| tensor-decoding stubs, run reused | 5,283,570 (full) | 1,222,380 |
| real scalar reference validators, whole preflight | 25.90M | 25.90M (unchanged) |

Findings: the executor's own loop is ~230 pcycles/command preflight and ~240
dispatch (about 1.2 ms per pass at 1.84 GHz), so executor overhead is not the
non-conv bottleneck. The real cost is the handlers' validators, ~5,100
pcycles/command (~26M pcycles, ~14 ms) per pass. Before, a Frame inference did
this three times (prepared_run preflight, run's own preflight, reference_run's
dry pass), ~78M pcycles (~42 ms, ~1.2%); now once, plus ~0.1M. Expected gain
~50M pcycles (~28 ms, <1%) per inference, not seconds. The one-command-per-run
cycles harness cannot show it (prefix totals 69.21M -> 69.49M primary,
38.35M -> 38.66M alternate, within 1%, since each command is its own preflight).
Plan-cache for the reference chain was not made real: the remaining overhead
does not justify it; the dominant 2.5 s is handler compute.

Evidence: host `test_prepared_executor` (new profile/reuse/no-effect test),
`test_nprg2_plan_cache`, ASan `test_prepared_elementwise_host`,
`test_prepared_full_model` primary/alternate fast and mixedfast (lockstep and
whole, outputs equal emulator sha256) pass; V75 `run_prepared_reference_sim.sh
both prefix 6 fast cycles` passes (all seven pools identical) for both models.
The engine with the Frame flags + elementwise + profile builds from a scratch
copy (`.work/engine-build`), and builds with `QDS_PREPARED_PROFILE=0`. Not run
on hardware. Hardware must check: `executor_profile.preflights_reused==1`,
`validated_commands==commands`, outputs still equal emulator sha256, and that
arena bytes 20480..24640 are unused by anything else.
