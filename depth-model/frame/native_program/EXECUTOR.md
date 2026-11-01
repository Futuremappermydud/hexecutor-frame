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
