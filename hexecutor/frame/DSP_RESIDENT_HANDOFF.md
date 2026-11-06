# Handoff: run the Model stereo graph entirely on Frame DSP

## Objective

Run both recovered Model stereo models as resident DSP graphs, with ARM limited
to input submission, completion and output retrieval. Preserve exact output
bytes against the Qualcomm HTP simulator for the existing probe suite. Measure
cold and sustained warm inference, and leave the existing point-cloud backend
working until the new path passes its gates.

This is **not implemented yet**. The current v5 package is hybrid: Python/ARM
executes the graph and sends 1,146 DSP requests per inference. It uses HMX for
916 of 1,036 convolution tiles, and HVX for the remainder; warm primary smoke
is 0.401–0.403 s. Both models passed 25 synthetic probes plus random input, with
exact output hashes. A live Model service reported roughly 10.5 ms disparity
compute and 14 ms core compute; those are device telemetry, not a speed
prediction for Frame.

## Locations and baseline

- Source: `/Users/owenelliott/Documents/projects/spatial-stuff/depth-model`.
- Frame development copy: `~/frame-dsp/depth-model/depth-test-20261029/work`.
- Frame existing usable package: `~/frame-dsp/depth-model/depth-model-direct-v5`.
- Point-cloud integration: `/Users/owenelliott/Documents/projects/frame-spatial-runtime`, `src/model_depth_pipeline.cpp` and `scripts/model_depth_bridge.py`.
- QAIRT SDK: `/Users/owenelliott/Downloads/qairt/2.50.0.260828`.
- Reference contexts: `models/{primary,alternate}/context.bin`.
- Graph metadata and execution order: `recovered/{primary,alternate}/graph.json.gz`, `execution_order.json`, `kernel_calls.json`, `constants/constant_pool_5.bin`.
- Existing correct implementations: `tools/htp_emu.py`, `frame/fast_depth.py`, `frame/qd_memory.c`, `frame/qd_conv.c`, `frame/qd_hmx_kernel.h`, `frame/qd_hmx_ops.S`, `frame/frame_transport.py`.
- Hardware acceptance: `frame/bench/verify_optimized.py --hmx --report REPORT.json`; simulator references in `reference/simulator_output_hashes.json` and `recovered/*/emulator_random.json`.
- Prior measurements: `frame/build/depth-hmx-u8-cached-primary.json`, `frame/build/depth-hmx-u8-cached-validation.json`, `frame/build/depth-v5-package-stereo.json`.

Frame alias is `ssh frame` (steamos). Before a hardware session, check for an
active point-cloud/NPU chat session, acquire the existing transport lock, and
inspect DSP state with `frame/bench/depth_state.py`. Do not interrupt another
client. The user explicitly supplied the recovery command if a reboot is
needed: `ssh frame "echo 'frame1' | sudo -S reboot"`. A reboot interrupts the
headset and camera stream; avoid it unless the DSP is actually stuck.

## First decision: existing graph runtime or custom executor

1. Inventory Frame's actual DSP firmware, registered services, FastRPC devices,
   loaded libraries and available QuRT entry points read-only. The SDK includes
   `lib/hexagon-v75/unsigned/libQnnHtpV75.so`, its skeleton and ARM64 host
   libraries. Both original contexts execute unchanged under QAIRT 2.50's x86
   HTP simulator without a custom op package. This does **not** establish V75
   compatibility or that Frame exposes the required runtime.
2. Attempt a bounded, isolated QNN context-load/one-inference experiment only
   if the service and loader path is demonstrably available. The existing Valve
   callback by itself is not a QNN/QuRT loader: the SDK HTP binary imports many
   `qurt_*`/`HAP_*` functions. Do not assume an Android skeleton works on the
   SteamOS host or change firmware/services to force it. Compare both outputs to
   smoke hashes, confirm teardown and capture exact failures. If no usable path
   exists, record the concrete blocker and use the custom path below.
3. Do not spend the entire project on an unproven runtime shim. A successful
   context execution with correct output is the gate for selecting that route.

## Custom resident-executor plan

1. **Generate a deterministic graph program offline.** Lower the recovered
   execution order, tensor descriptors, layouts, constants and recorded kernel
   arguments into a versioned binary command stream. Preserve producer order,
   aliases, padding, zero-fill and checkpoint dependencies. Do not treat every
   `@SyncOp` as meaningful computation without checking the simulator schedule.
   The primary graph has 5,068 scheduled nodes across 75 type names; alternate
   has 5,040/75. Compare generated descriptors and dependencies with the Python
   graph for both models and reject missing/unknown operations explicitly.
2. **Prove a safe memory layout before running a full graph.** The original
   primary uses an 8,388,608-byte runtime pool, a 7,209,216-byte constant pool,
   1,769,472-byte spill area and two small pools; alternate is similar. The
   current callback arena is only 8 MiB and its HMX scratch is 0x2d0000 bytes,
   so these pools cannot simply be put together there. Inventory the actual DSP
   mappings and ownership; use only verified free ranges, or implement a tiled
   allocator/staging protocol. Never assume the whole 32 MiB service buffer or
   all 8 MiB VTCM is free. Preserve immutable constants, tensor aliasing and
   aligned HMX activation/result/completion buffers. Check uncached ordering,
   bounds, scratch reuse and ownership with canaries and repeat sessions.
3. **Run a resident command list.** Extend the existing sequence-tagged worker
   protocol with a versioned graph command and bounded worker barriers. Upload
   graph/weights once, send inputs per frame, return only output heads and a
   diagnostic status. Reuse the exact loader ELF: its SHA-256 is
   `828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572`;
   do not rebuild/replace it. The rewritable payload may change between sessions.
   Retain cancellation, idle timeout, eye-client handling and the saved AXI/HMX
   power state until all workers have exited and hardware state is restored.
4. **Port kernels in graph order and validate at boundaries.** First run one
   representative convolution followed by a blocked-tensor write/read on DSP,
   then a multi-convolution residual block (including add/requant), then the
   full uint8 backbone. Existing exact HMX code provides the convolution and
   quantized epilogue; implement DSP-side tensor packing/layout, elementwise
   operations, resize, stride-2/space-to-depth, transpose, slice/pad, concat,
   gather and the uint16 branch. Preserve saturation, shifts, rounding and
   overflow as in the reference; explicitly handle dynamic weights/Fi and
   output-head floating point. The original has 110 uint16 convolutions and
   currently splits each into two byte-plane requests; retain exactness before
   optimizing their DSP implementation. Avoid interpreting model outputs as
   confidence or changing temporal semantics as part of this performance work.
5. **Localize mismatches.** Add optional per-node or per-block checksums and
   snapshots at graph boundaries, compared with `htp_emu.py` / QNN simulator;
   keep diagnostics disabled for timing. Exercise both zero/identity smoke and
   nonzero random hidden/transform inputs so temporal and uint16 paths run.
   Avoid caching runtime-produced tensors or dynamic weights as constants.
6. **Integrate only after correctness.** Provide a persistent backend API that
   accepts native x, y, hidden and t and returns both full output tensors.
   Maintain v5 as fallback; add explicit backend selection to the point-cloud
   bridge. Leave its capture-time pose and experimental confidence/geometry
   assumptions unchanged until independently validated.

## Acceptance and measurements

- Run all 25 recorded stereo probes plus seed-1234 random inputs for **each**
  graph, with exact hashes for both heads; validate repeated sessions and error
  recovery. Include targeted tests for odd channel counts, 16-bit boundary
  values, HMX K depth >32 tiles, padding and aliased storage.
- Check real Frame startup, sustained repeated inferences, cancellation and
  restart. Report cold and warm latency, per-stage DSP time, bytes moved, request
  count, memory footprint and effective inference FPS. Time inference separately
  from model loading, camera capture, rectification, rendering and cleanup.
- Confirm DSP controls `[1,0,1]`, callback zero, loader/payload/arena restored,
  AXI and HMX state restored, and the eye client restored after both success and
  failures. Keep the existing chat installation and other active clients intact.
- Test the point-cloud image bridge using a recorded pair, compare the output
  with v5, and measure rendering separately from depth cadence.
- Deliver a new versioned package and update `frame/DIRECT.md` with exact tested
  numbers and limitations. Do not describe the full graph as DSP-resident until
  all executed graph operations actually run there; only host input/output,
  setup and control are allowed by this goal.

## Model geometry is a separate workstream

Live Scene Setup reported RNN motion compensation enabled, separate left/right
history, pyramid level 1, uncertainty maps and active surface fusion. The v5
Frame app uses zero history and provisional plane-0 disparity with independent
census checks. Running the graph on DSP will not by itself fix floating points,
recover Model's temporal warp/confidence semantics, or establish metric accuracy.
See `README.md` and `metadata/model-runtime/native-code-evidence.json`.

## Earlier checkpoint (Oct 29, 2026, 22:20 EDT)

`resident_program.py` now compiles both recovered graphs to a deterministic,
versioned QDRP offline IR, with stable operation IDs, full relative tensor
descriptors and aliases, recorded kernel arguments, retained checkpoints and
conservative serial command dependencies. It checks source context identities,
unknown handlers, missing tensor references, required arithmetic arguments,
pool bounds, spill/fill ranges, block-zap ranges, binary integrity and exact
source round trips. Duplicate InputSlicePad entries are explicit no-ops on
their second occurrence (25 primary, 19 alternate). All 75 scheduled operation
types are represented: 5,068 primary commands and 5,040 alternate commands.
Eleven integrity/negative/real-graph tests pass. This does not validate DSP
numerical execution or establish a working resident backend.

Run from the source root:

```sh
python3 frame/resident_program.py recovered/primary --output frame/build/resident-primary.qdrp
python3 frame/resident_program.py recovered/alternate --output frame/build/resident-alternate.qdrp
python3 frame/tests/test_resident_program.py
```

The IR intentionally retains source metadata for auditing; the current files
are 31,279,510 and 30,058,243 bytes. They are NOT the compact DSP upload format.
Next: intern shared tensor block tables and lower fixed-width DSP descriptors
and kernel immediates, then implement a bounded DSP decoder/dispatcher.
No DSP reader or new transport command has been implemented.

The original pools require 18,146,816 bytes (primary) or 18,234,112 bytes
(alternate), excluding IR, input/output staging and 2,949,120-byte HMX scratch.
They exceed the verified 8 MiB arena by 9,758,208 / 9,845,504 bytes. Reports
set `hardware_layout_verified: false`; no new address range was claimed.
A tiled allocator or additional proven ownership is still required.

Read-only Frame inventory found Valve `dsp_service` with the recognized
firmware SHA-256 `4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c`,
`/dev/cdsp` and `/dev/adsp`, no FastRPC device, no QNN/FastRPC runtime in the
searched `/opt/steamvr`, `/usr/lib`, `/lib/firmware`, `~/frame-dsp` trees, and
only `tms/servreg` in the installed CDSP service registry. `/proc/1006/maps`
was inaccessible to the ordinary user. This is a concrete missing host/service
route, not proof that QNN could never run with different firmware. No context
load was attempted, no firmware/service configuration was changed.

The first two locked state snapshots showed identical orphaned callback
`0x87606000`, controls `[2,0,8]`, nonzero regions, stopped eye client and no
lock holder/inference process. The handoff-authorized reboot recovered Frame.
Final locked snapshot: controls `[1,0,1]`, callback zero, all loader/payload/arena
regions zero, eye client PID 2924 in `S`. Existing loader ELF, payload, packages
and point-cloud integration were not modified. Raw final inventory is in
`frame/build/resident-runtime-inventory.txt`.

The remaining custom-executor, memory, exact-output, latency, recovery and
point-cloud acceptance gates above are still outstanding. v5 remains the
usable backend; there is no v6 package or new inference-performance claim.

## Latest continuation checkpoint (Oct 29, 2026, 22:38 EDT)

The next lowering/reader milestone is implemented and tested. Full resident
inference is still not implemented. Start with `frame/RESIDENT_FORMAT.md` for
the wire ABI, commands, results and precise scope.

New code:

- `frame/resident_image.py` lowers audited QDRP into deterministic QDRX v1.
- `frame/resident_schema.py` and generated `frame/qd_resident_schema.h` define
  exact kernel variants/call IDs and ABI fingerprint `beb0897a`.
- `frame/qd_resident.{c,h}` provide a freestanding allocation-free decoder and
  single-worker serial dispatcher core. Every required variant and the command
  budget are checked before any kernel callback; only explicit NOPs are built
  in. Cancellation and kernel-error reporting are tested on the host. There
  are no numerical graph kernels, mapped-pool allocator or persistent worker
  graph protocol yet.
- `frame/bench/qd_resident_probe.c` and `verify_resident_decoder.py` exercise
  the native reader on Frame without allocating graph pools or executing graph
  kernels. The probe does not touch AXI/HMX state or VTCM.
- `frame/tests/test_resident_image.py` has 13 passing semantic, native-boundary
  and dispatcher tests. The original 11 QDRP tests still pass. The ASan/UBSan
  driver `frame/tests/fuzz_resident_decoder.c` passed 512 checksummed mutations
  and dispatch attempts per graph, including unaligned image bases.

Generated outputs:

| Model | Image | Bytes | SHA256 |
| --- | --- | ---: | --- |
| primary | `frame/build/resident-primary.qdrx` | 1,738,088 | `dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b` |
| alternate | `frame/build/resident-alternate.qdrx` | 1,736,920 | `54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298` |

Each image has a companion JSON manifest. Exact shared block tables, fixed
128-byte tensor descriptors, 96-byte commands and 88-byte recorded-call records
replace the bulky source metadata. All tensor storage/layout/quantization/
scalar metadata and every command/checkpoint/framework vector/call immediate
are reconstructed and compared with the source, including both graph heads
and dynamic weight/Fi aliases. No physical DSP addresses are assigned. The
reader checks CRC, ABI, canonical sections, bounds, relative pools, arithmetic
call presence, serial dependencies and spill/fill/zap spans. Aggregate work is
bounded even when an input reuses large shared spans many times.

Cross-build succeeded using the existing `depth-model-hvx-build` Docker image.
`frame/build/qd_resident_probe.elf` is 9,176 bytes of text/rodata, has no runtime
imports and SHA256 `b24084b56ac864558375b8e32366221f8f09eded175c4258ec8be48c5040c396`.
Only the rewritable payload was built/installed; the exact loader ELF was not
rebuilt or replaced.

Before hardware work, read-only process/lock checks found no active depth/NPU
client. A locked `depth_state.py` snapshot showed controls `[1,0,1]`, zero
callback and zero loader/payload/arena regions, eye client PID 2924 in `S`.
The decoder probe passed **28 cases across two separate transport sessions**:
seven cases per graph (valid, missing-kernel, budget, CRC, tensor-ref, block-range
and truncation). Both source graph descriptor counts matched on DSP. Failures
returned the expected codes without graph-kernel execution. Results and locked
before/after snapshots are in `frame/build/resident-decoder-hardware.json`, also
present in the Frame development copy.

Valid-image decoder checks took 3.3846–3.3888 seconds wall time through the
uncached arena, including command upload/completion. These are unoptimized
setup-validation timings, NOT inference latency. The full-graph latency/FPS
acceptance gates have not been run on this path.

After each session controls were `[1,0,1]`, callback zero and loader/payload/
arena zero. Final eye client PID 12170 was in `S`. A separate locked final
snapshot confirmed the same state. No reboot, firmware/service changes, v5
package changes or point-cloud integration changes were needed.

The memory requirements in the earlier checkpoint are unchanged, and the QDRX
image itself also needs staging/storage. `hardware_layout_verified` is still
false. Next: establish a safe pool/staging/tiled-allocation plan, implement
bounded tensor access and per-kernel numerical preflight, then prove a single
convolution plus blocked write/read before expanding to a residual block.
Persistent multi-worker graph publication/barriers and hardware cancellation
remain separate required work. Keep v5 usable until all original exact-output,
memory, recovery, latency and bridge acceptance gates pass.

## Model-level performance feasibility

Model's observed service telemetry was about 10.5 ms disparity compute and
14.3 ms core compute (`README.md`, Live Model observations). These are stage
timings, not independently measured end-to-end latency. Matching those numbers
from the current approximately 402 ms warm Frame inference would require about
28–38x overall speedup, with equivalent workload and timing boundaries confirmed.

The engineering assessment is that parity is plausible enough to investigate,
but remains an ambitious, unproven target. Full DSP residency alone does not
establish it. The existing hardware profile provides the following evidence:

- `frame/build/depth-hmx-u8-cached-primary.json`, warm run 2: 402.115 ms total,
  117.239 ms waiting for DSP requests and 284.877 ms outside that wait. The
  latter includes host graph work, preparation, collection and transport
  bookkeeping. The wait includes DSP computation, copies, synchronization and
  polling/transport effects; it is not a pure compute measurement or a hardware
  lower bound. These categories describe the current implementation, not the
  latency of a future resident graph.
- The same report records aggregate HMX phase counters across five inferences
  (one cold, four warm). Dividing by five and the measured 1,843.2 MHz DSP core
  clock gives approximately 15.53 ms copy/zero/barrier, 44.94 ms activation
  preparation/barrier, 10.17 ms calculation/synchronization and 10.95 ms
  quantization/store per inference. Definitions are in `qd_conv.c::hmx_conv`;
  clock calibration is documented in `frame/HMX.md`. Per-phase counters sum
  the maximum worker duration per request, so these are diagnostic estimates.
- Those HMX phase numbers cover only the 916 HMX convolution tiles, not all
  1,036 convolution tiles or the rest of the graph. The uint16/HVX paths and
  other operations still need their own optimized resident implementations.

There is substantial measured overhead to target through resident execution,
better activation packing, tensor/weight reuse, less memory movement and kernel
fusion. Removing host work while retaining the existing DSP data path would
still leave a large gap to the Model stage timings. The current HMX calculation
interval alone already uses roughly 10 ms; matching Model requires improvements
throughout the data path while preserving exact numerical results.

The next performance decision gate is a resident convolution plus blocked
write/read, followed by a representative residual block, measured with actual
graph shapes, bytes moved and separate packing/compute/store timings. Use
coverage-weighted measurements of both uint8 and uint16 paths to estimate the
full graph before promising a latency or FPS target. Full-model sustained
timings, exact-output checks and equivalent Model measurement boundaries remain
the acceptance criteria. No new performance claim follows from the decoder-only
probe or its setup-validation time.

## Native executor checkpoint (Oct 30, 2026)

Whole-graph execution moved from Python into C (`frame/qd_graph.c`, driven by
`frame/native_graph.py` over the QDRX image). ARM runs every non-conv op in C;
convolutions go to a revised rewritable payload (`qd_conv.c`; loader ELF
unchanged, `828ca063…`). This is still hybrid ARM+DSP, not DSP-resident.

Acceptance (`frame/bench/verify_native_graph.py --backend dsp --repeat 20
--helpers 6,5,4`, report `frame/build/native-graph-dsp-validation.json`):
all 25 probes + smoke + seed-1234 random are bit-exact for both models (54/54),
and all 40 sustained warm runs are exact. Warm latency 120.6–122.9 ms (median
122.0 primary / 121.8 alternate), first inference ~170 ms, versus 401–403 ms for
v5. After the run: controls `[1,0,1]`, callback zero, regions zero, eye client restored.

What changed, in order of effect (warm primary, ms): C executor 402→283;
pipelined double-buffered DSP requests with 2 KB-block hazard footprints 283→248;
C kernel fixes 248→222; arena-copy helper threads (arena reads are ~2 GB/s per
core, ~9 GB/s across 4) 222→202; scatter-free HMX activation prep 202→179;
kernel specialization 179→170; fused single-request u16 conv epilogue on DSP
plus HMX weight residency in VTCM 170→157; 32-lane activation packing for HMX
157→149; exact access-box footprints + conv hoisting 149→131; parallel blocked
transfers 131→122. DSP-side changes were proven byte-identical to the original
scatter/two-plane code in the v75 ISA simulator (`bench/prepare_sim.c`,
`bench/u16_sim.c`) before hardware.

Remaining per-inference budget: DSP compute ≈ HMX 8 copy + 21 prep + 10 calc +
11 quant ms, HVX ≈ 13 ms; ARM ≈ 43 ms non-conv kernels plus conv glue. Helper
threads spin on cores 4–6 during inference; set OPENBLAS_NUM_THREADS=1.
Not yet done: v6 package, point-cloud bridge backend selection, DIRECT.md
user-facing update. v5 remains the packaged backend.

Continued in [NATIVE_EXECUTOR_HANDOFF.md](NATIVE_EXECUTOR_HANDOFF.md).
