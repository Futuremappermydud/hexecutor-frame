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
It copies those bytes directly after preflight checks the active pool-1 limit
and rejects input/output overlap.

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

`prepared_elementwise.c/.h` registers QDR ADD_U8 variant 13 and LinearClip
variant 68 (`linearclip@CH.CB.fi`, blocked CB u8 to CH u16). Other variants
remain uncovered. Both handlers read current bound tensor locations and
current Fi values; neither allocates or creates a dense intermediate. They
validate the exact blocked geometry, command coefficients or Fi shape/range,
pool-1 active write ceiling, touched-page bounds, and output/input page alias
rules before execution.

Alias proof uses caller-owned page-address arrays sized for the actual
maximum 2,560 block entries per tensor. The requirement is
`3 * 2560 * sizeof(uintptr_t) + 7 * sizeof(uint8_t *)` bytes: 61,464 bytes on
64-bit hosts and 30,748 bytes on Hexagon. The trailing seven pointers hold the
pool map needed by the audited HVX helpers. Preflight does no graph-pool
writes. The runner must reserve this scratch for command ranges containing
either handler. The current executor repeats whole-range preflight at run
entry; a later prepared-plan cache can retain the validated alias proof to
avoid repeating it across frame executions.

The fixture generator runs NativeGraph's CPU backend to the first command of
each registered family, snapshots every graph pool and the live LinearClip Fi,
executes that exact CPU command, and captures the physical output pages. The
V75 simulator harness reloads those pools, dispatches the same prepared
command, and compares every pool byte with the captured CPU step. It also
mutates a touched output page to alias an input and places an output page in
the reserved VTCM tail; both are rejected before any pool write. Numerical
execution calls the existing audited `blocked_glue/add_u8.c` and
`linearclip_u8_u16.c` implementations. This is a simulator proof, not a
device latency result.

`run_prepared_elementwise_host_sanitized.sh` runs ASan/UBSan read-only
preflight against actual LinearClip metadata from both NPRG images. Its host
link stubs cover only the unused numeric apply calls; V75 simulation above
executes the real HVX glue.
