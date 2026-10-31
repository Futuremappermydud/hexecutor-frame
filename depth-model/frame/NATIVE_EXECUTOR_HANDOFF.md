# Handoff: native executor and the path to Model-level latency

Continues `DSP_RESIDENT_HANDOFF.md` (its acceptance gates, safety rules and
Frame procedures still apply). Written Oct 30, 2026.

**Latest continuation:** see “Native integer HMX and resident diagnosis” below
and `RESIDENT_EXECUTOR_WIP.md`. **Before a reboot, the DSP mis-mapped parts of the
cdsp_mem window and the resident backend failed. After the reboot it passes cold on both
models. Run `bench/chunk_map.py` after every insmod. See the top of `RESIDENT_EXECUTOR_WIP.md`.** Current experimental hybrid medians are about
100 ms; installed v5 is unchanged. The older timings and “unimplemented”
resident status below describe earlier checkpoints.

## State in one paragraph

Both recovered stereo graphs now run end to end in a native C executor on ARM
(`frame/qd_graph.c`), with every convolution dispatched from C to a revised
DSP payload (`frame/qd_conv.c`). **Warm inference: 120.6–122.9 ms (median 122.0
primary / 121.8 alternate), down from 401–403 ms for v5; first inference ~170 ms.**
All 25 simulator probes + smoke + seed-1234 random are bit-exact for both
models (54/54), and so are 40 sustained warm runs. Report:
`frame/build/native-graph-dsp-validation.json`. This is still a hybrid ARM+DSP
design, **not** the DSP-resident goal. v5 remains the packaged, user-facing backend.

## Where things are

The source of truth is the Mac tree `/Users/owenelliott/Documents/projects/spatial-stuff/depth-model`.
**It is not a git repo.** v5 copies of the changed payload files are in
`frame/build/{qd_conv.c,qd_conv.h,qd_hmx_kernel.h}.v5`.

Frame dev copy: `~/frame-dsp/depth-model/perf-20261029` (a copy of the v5 package
with the new files scp'd in). It is in sync with the Mac as of this handoff.
After editing on the Mac, copy the changed files there. The `.so` rebuilds
itself on first use.

| File | Role |
| --- | --- |
| `frame/qd_graph.c/.h` | Executor: QDRX decode, all 44 opcodes / 75 variants in C, CPU and DSP conv backends, pipelining, hazard footprints, hoisting, helper threads |
| `frame/native_graph.py` | ctypes wrapper. Owns pools; `use_cpu()` / `use_dsp(transport)` / `set_helpers([cores])` / `set_hoist()` / `run(inputs)` |
| `frame/bench/verify_native_graph.py` | Acceptance + timing (`--backend cpu|dsp --helpers 6,5,4 --repeat N --profile --step CASE --report`) |
| `frame/qd_conv.c`, `qd_conv.h`, `qd_hmx_kernel.h` | Payload. New flags `QDC_U16=8`, `QDC_WEIGHTS_RESIDENT=16`, `QDC_X32=32`, plus a scatter-free prepare |
| `frame/bench/prepare_sim.c`, `u16_sim.c` | v75 ISA-simulator equivalence tests for the payload changes |
| `frame/build/resident-{primary,alternate}.qdrx` | Graph images (rebuild with `resident_image.py` if missing) |

At this checkpoint, payload `build/qd_conv.elf` was SHA-256 `8644c4fdd35896b1d2f5e922d9c3bdcf9a6b84d9ec347c6b04b4599624a7228c`
(17,968 bytes of text/rodata within the 32 KB limit). **Stale:** the on-disk file is now
`b57c41a8a091717b10085fe442e331a9717e963eb50a6ca8d14a181fb68d349c` (modified Oct 30 02:43,
provenance not recorded); the native experiments use `qd_conv_native.elf` instead. The loader is unchanged at
`828ca063…` and must stay that way. The earlier payload and sources are saved in
`frame/build/pre-direct-collect/`; see the continuation below.

## How to run

```sh
# Mac, no hardware (~2 min per model; exact vs. reference hashes)
.venv-cpu/bin/python frame/bench/verify_native_graph.py --backend cpu
# localize a mismatch: compare every pool with the Python emulator after each command
.venv-cpu/bin/python frame/bench/verify_native_graph.py --backend cpu --model primary --step smoke

# Frame (check for other DSP clients first; the transport takes the lock)
cd ~/frame-dsp/depth-model/perf-20261029
~/frame-dsp/depth-model/.venv/bin/python frame/bench/verify_native_graph.py \
  --backend dsp --repeat 20 --helpers 6,5,4 --report frame/build/native-graph-dsp-validation.json
~/frame-dsp/depth-model/.venv/bin/python frame/bench/depth_state.py   # expect [1,0,1], callback 0, regions zero

# Payload build (Mac; never rebuilds the loader)
cd frame && docker run --rm -v $PWD:/src -w /src depth-model-hvx-build:latest \
  make HEX_CC=clang-14 HEX_LD=ld.lld-14 build/qd_conv.elf
# ISA simulator (SDK 6.6 under .work/hexagon-sdk; x86 image under emulation)
SDK=$PWD/../.work/hexagon-sdk/6.6.0.0; T=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin
docker run --rm --platform linux/amd64 -v $SDK:/sdk:ro -v $PWD:/src -w /src depth-model-hmx-tools:latest \
  sh -c "$T/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -G0 bench/prepare_sim.c -o build/prepare_sim.elf && $T/hexagon-sim -mv75 --quiet build/prepare_sim.elf"
# u16_sim.c additionally needs qd_hmx_ops_encoded.S on the command line
```

`--profile` prints per-variant ARM time, hazard drains, and the hoisting outcome
counts. Each timing record carries DSP phase cycles (`hmx_phase_cycles`,
`hvx_phase_cycles`) at 1843.2 MHz.

## Design invariants (break these and outputs go wrong silently)

- **Kernels mirror `tools/htp_emu.py` / `fast_depth.py` exactly**, including NumPy
  broadcasting, crouton lane extension on write, float rounding (`-ffp-contract=off`,
  explicit `volatile` float steps) and argument validation. Unsupported shapes or
  arguments return an error; they are never approximated.
- **Hazard footprints** (`build_footprints`): each command's set of 2 KB blocks.
  The footprint is the exact box each kernel reads (dims at padding, or the full
  extent for copy_activation/space_to_depth) and writes (lane-extended), the
  exact conv read/write boxes from `conv_boxes()`, and explicit ranges for
  spill/fill/zap/convert_weights. **If you add or change a kernel's access
  pattern, update its footprint rule**, otherwise pipelining reads stale data.
- **Pipelining**: at most one DSP request runs while the next is packed into the
  other arena half. A command whose footprint hits the in-flight conv's output
  either hoists a later independent conv (window 64) or drains. Convs finalize in FIFO order.
- **Weight residency**: keys are globally unique (static atomic). The host mirror
  resets at every `qdg_run` and whenever workers are not persistent. The DSP
  refuses `QDC_WEIGHTS_RESIDENT` unless its own record matches, and any non-HMX
  command clears it.
- **u16 single request** is exact by `floor((256·hi+lo+corr)/256) = hi+ch+floor((lo+cl)/256)`.
  Eligibility (|acc|, |ch| < 2^29, post 1..15) is checked per channel; otherwise
  the old two-plane path is used (10 of 110 convs).
- **x32 / fast prepare** depend on the host zeroing the 128 bytes after the
  activation packet, and on `ci % 32 == 0` for x32.
- **Constant pool 5 is never written** (verified for both graphs); the weight and
  quant caches rely on that.
- **Payload compatibility**: the new payload still accepts v6 packets from the
  Python `FrameTransport` path (new flags default off), but the v5 Python path
  was subsequently re-validated with the current payload (52 probe/random cases;
  see the continuation below). The installed v5 package remains unchanged.

## Operational gotchas (all hit this session)

- ARM reads of the dsp_service dma-buf arena are uncached (~2 GB/s per core,
  ~9 GB/s across 4); writes are fast. That is why the helper threads exist.
- HVX arrays that spill to the stack land in DDR, because L2 is TCM. A 2x
  slowdown came from `HVX_Vector a[8]`; keep vectors in named registers and
  check the disassembly (`llvm-objdump-14 --mattr=+hvxv68,+hvx-length128b`).
- The simulator's `upcycle`/`pcycle` read 0; measure cycles on hardware.
- Set `OPENBLAS_NUM_THREADS=1`: numpy's BLAS workers spin on the cores the helpers use.
- **A hang inside C cannot be stopped by SIGTERM** (the Python handler never runs).
  ptrace is now allowed on Frame, so use `gdb -p PID -batch -ex "thread apply all bt"`.
  Recovery without reboot, as done this session:
  1. Confirm controls `[1,0,1]`, zero worker status, and AXI/HMX markers 0 (read-only).
  2. `kill -9` the process.
  3. Set WORK+24 to 0, then zero the loader (0x606000, 4 KB), payload (0x610000, 32 KB) and arena (0x1800000, 8 MB).
  4. SIGTERM+SIGCONT the paused eye client, and relaunch it if it is not respawned within ~0.4 s.

  If any marker is nonzero or the controls differ, use the reboot from the main handoff instead.
- The hang itself was a helper-pool startup race (now fixed: helpers start from
  generation 0). Keep any future pool changes covered by a two-model back-to-back run.

## Where the remaining 122 ms goes

| Component | ms/inference | Notes |
| --- | ---: | --- |
| HMX calc | ~10 | Already ≈ Model's 10.5 ms disparity stage |
| HMX input prep (im2col into VTCM) | ~21 | 9× activation expansion for 3×3, u8 in 16-bit lanes |
| HMX quantize + copy out | ~11 | |
| HMX copy in | ~8 | 66 MB activations, 25 MB weights (after residency) |
| HVX convs (u16, odd shapes) | ~13 | |
| ARM non-conv kernels | ~43 | Elementwise, requant, transpose, resize, layout |
| ARM conv glue/collect | ~20 | Partly overlapped |

Dependency chains (conv → add → conv) serialize ARM and DSP, so the total is
more than either side's busy time.

## Assessment: blockers to Model-level latency

Model telemetry is ~10.5 ms disparity / ~14.3 ms core. Workload and timing
boundaries are still **unconfirmed** (resolution, pyramid level, what "core"
includes); confirm them before treating parity as the target.

1. **HMX use.** We build explicit im2col tiles. QNN almost certainly uses HMX's
   native activation addressing and fused output conversion. This accounts for
   ~30 ms of DSP time and needs more of the recovered HMX encoding (research).
2. **No residency.** Every tile's output goes DSP → uncached arena → ARM →
   re-layout → DSP (~130 MB/inference). Fixing it needs the DSP-resident
   executor. The concrete blocker is memory: pools need ~18 MB, but only the
   8 MB arena is verified. Either prove more ownable memory or tile layers to fit VTCM.
3. **ARM runs the glue ops** in the dependency chain (~43 ms). Porting them to
   HVX is straightforward but large (~40 kernels kept bit-exact). The C kernels
   in `qd_graph.c` are the reference to port from; the QDRX decoder already runs on the DSP.

Without (2), I expect a floor of roughly 50–60 ms.

## Suggested next steps

1. **Quick win (~10–15 ms):** fuse the HMX quantize with writing results in
   the consumer's crouton layout, and copy whole blocks instead of the
   staging → dense → blocked passes.
2. Further ARM kernel work on the top `--profile` entries (Add, requant,
   linearclip, transpose, im2col), plus parallel compute loops on the helper pool.
3. Integration (previous handoff, step 6), then a v6 package:
   - a persistent backend in `native_graph.py` for the point-cloud bridge, with explicit selection between it and v5;
   - helper cores configurable and off by default if the runtime needs those cores;
   - `DIRECT.md` updated with the numbers above.
4. Residency groundwork: a DSP memory-ownership inventory beyond the arena, or a
   per-layer VTCM tiling plan, then port glue kernels to HVX starting with the
   most frequent ones (Add.tcm, linearclip_simple_rq, SpaceToDepth, ResizeNear).
5. HMX native-convolution research, in parallel, against the v75 simulator.

## Continuation: direct crouton output (Oct 30, 2026)

The proposed output-layout optimization is implemented as an **opt-in
experiment**, not a new performance default. The source tree and Frame
`perf-20261029` development copy contain the current code and payload. v5 is
still the installed user-facing package; there is no v6 package or point-cloud
integration change, and full DSP residency is still unimplemented.

### Implementation

- `QDC_CROUTON8=64` extends protocol v6 with logical output height in flag bits
  8..15 and width in bits 16..31. Both must be positive multiples of 8; shape
  validation rejects an output larger than the existing padded C allocation.
  No new DSP memory range is used. Legacy flags default off.
- HMX quantization writes directly to each worker's own arena output. The
  previous VTCM quantized-output copy is removed in both modes. This also
  removes a possible cross-worker read-before-quantization-completes hazard
  in the old shared copy phase; no failure from that hazard was observed.
- Block mode emits `[h/8][w/8][n/32][8][8][32]` and collects complete 2 KB
  croutons directly into ARM graph pools. This avoids staging and dense
  intermediates for 889 of 916 HMX requests. Unaligned outputs use direct
  eight-pixel-strip collection or the existing fallback. All writes retain
  the existing convolution footprint and FIFO completion order.
- Parallel block copies start at 16 KB. Their independence is checked once
  per immutable command output box; duplicate destination blocks retain
  serial write order. Partial-strip copies use a conservative whole-table
  uniqueness check before using helpers.
- Enable with `NativeGraph.set_block_output(True)` or verifier
  `--block-output`. **Default is off.** This requires the current payload;
  do not combine a block-mode host with an older payload. There is no payload
  feature negotiation. `Stats.hmx_block_calls` reports actual use.

### Results

Two full-suite sessions used CPU 7 and helper cores 6,5,4. Each mode passed all
27 cases per model (smoke + 25 probes + seed-1234 random), plus 20 sustained
warm smoke runs per model. Both heads matched exact reference hashes.

| Mode | Primary median (min–max), ms | Alternate median (min–max), ms |
| --- | ---: | ---: |
| Default native | 122.545 (121.007–124.396) | 122.239 (120.712–123.483) |
| Experimental block output | 121.824 (120.319–125.604) | 121.164 (120.123–122.267) |

First smoke was 168.650/164.345 ms default and 163.798/161.818 ms block mode.
These are inference timings, excluding graph creation and session setup.
Median collect time falls from ~16.2 to ~13.3 ms, but median DSP wait rises from
~51 to ~56 ms as less ARM work overlaps DSP execution. DSP epilogue time also
rises slightly. **The suggested 10–15 ms end-to-end win did not materialize**;
the measured median improvement is only 0.7–1.1 ms, with overlapping ranges.
Treat this as a validated layout building block for residency, not a major
latency advance. Timing categories overlap and must not be added together.

The experimental HMX output writes 46,116,864 arena bytes per inference versus
46,878,720 in legacy layout. Arena allocation and safe memory-ownership limits
are unchanged. There are still 1,046 DSP requests per native inference.

Reports (also copied back to the Mac):

- `frame/build/native-graph-default-validation.json`
- `frame/build/native-graph-block-validation.json`
- `frame/build/depth-hmx-payload-compatibility.json`: older Python/HMX path,
  all 25 probes + random for both models, **52/52 exact** with the new payload.
- `frame/build/native-graph-block-no-helpers.json`: smoke/random and two warm
  repetitions per model, all exact. The helper-free timings varied greatly
  (135–256 ms warm in this session); they are not a stable performance claim.
- `frame/build/native-block-{before,final}-state.json`: locked snapshots.
  Final controls `[1,0,1]`, callback zero, loader/payload/arena zero, eye client
  PID 52806 in `S`. Transport teardown also restored clock/power state. No
  reboot or firmware/service changes were needed.

Local checks: `tests/test_native_collect.c` passes 68 ASan/UBSan cases, including
partial tiles, flattened 1x1 output, nonzero origins, slices, holes, shuffled
pools, repeated destination blocks, padding canaries and helpers on/off. The
v75 `bench/hmx_conv_sim.c` passes five independent int64 convolution oracles
and 14 block-layout variants, including width 24 and DDR output canaries.
Log: `frame/build/hmx-conv-direct-sim.log`. Publication, native packet and six
host quantized-convolution random/boundary checks also pass.

Reproduce hardware validation:

```sh
cd ~/frame-dsp/depth-model/perf-20261029
OPENBLAS_NUM_THREADS=1 ~/frame-dsp/depth-model/.venv/bin/python \
  frame/bench/verify_native_graph.py --backend dsp --block-output \
  --helpers 6,5,4 --repeat 20 --profile \
  --report frame/build/native-graph-block-validation.json
```

The useful next work is outside this output-copy experiment: integrate the
persistent native backend with explicit v5 fallback, or address the residency
memory blocker and dependency-chain glue kernels. Keep block output opt-in
unless further profiling establishes a meaningful end-to-end benefit. The
original memory, cancellation/recovery and point-cloud acceptance gates remain.

## Native integer HMX and resident diagnosis (Oct 30, later continuation)

The user explicitly authorized benchmarking native kernels with numerical
differences while retaining the exact backend, and asked to ignore eye tracking.
Do not ask for this authorization again. Restore callback, ownership and clocks
after experiments. No loader, firmware, installed v5 or Qwen changes were made. A private `cdsp_mem`
kernel module was loaded and DSP JTLB entries 64..71 rewritten; both were undone.

### Validated hybrid result

`build_native_hmx.sh` builds a separate SDK-compiled `qd_conv_native.elf`.
SHA-256: `a21ed93fb980450c636f05adc09c2d2bce8e4e1af8a5d248b218f46712ec8041`.
It accepts exact v6, native integer v7, and native crouton load/store v8. Native
modes affect HMX only; non-HMX and EXIT remain v6. NativeGraph's default remains
exact (`set_native_hmx(0)`); modes 1 and 2 explicitly select v7/v8. Block output
is still opt-in with `set_block_output(True)`.

Native HMX processes 64 u8 pixels per activation tile, uses native `cvt.ub`, and
preserves the original arena ABI. V8 uses `:deep:cm` loads and `:cm` stores.
Weight-residency addresses account for the new tile count; primary/alternate
reuse 421/389 weight uploads. No new shared-memory region is needed.

All 54 input cases ran A/B/C/A2 benchmarks: exact A/A2 match reference
hashes; native B/C are not reference-equal (`equal: False` in all 54), and v8
matches v7 byte for byte. One warm run per phase/case, helpers
6/5/4, main CPU 7, OPENBLAS_NUM_THREADS=1:

| Mode | Primary median | Alternate median |
| --- | ---: | ---: |
| Exact, ordinary arena output | 115.54 ms | 116.06 ms |
| Native v7, ordinary output | 105.38 ms | 106.39 ms |
| Native v8, ordinary output | 101.38 ms | 102.97 ms |
| Exact, block output | 114.50 ms | 114.89 ms |
| Native v7, block output | 103.73 ms | 104.67 ms |
| Native v8, block output | 99.62 ms | 100.38 ms |

Reports: `native-hmx-cm-full-ab.json` and `native-hmx-cm-block-full-ab.json`.
Five repetitions on smoke/random with block output gave 98.4–99.4 ms; do not
substitute those selected-case numbers for the complete-suite medians.
All timing numbers exclude setup. Profile categories overlap.
Every phase, exact included, uses `qd_conv_native.elf`. Exact on that payload
(~115 ms) is ~7 ms faster than the earlier `qd_conv.elf` exact run (~122 ms),
unexplained; native arithmetic accounts for ~15 ms (115 → 100), not 22.

Native arithmetic is **not output-equivalent to the recovered simulator**.
Per-convolution independent ISA tests bound the conversion delta to 0/+1,
but this propagates through the graph. Final u8 head: smoke mean absolute
difference 2.0703 levels, p95 6, max 15; random mean 1.1010, p95 3, max 17.
The synthetic `w-2` case reaches max 98 levels (mean 2.2132). Float head random
MAE ~0.085 and max ~5.55; worst suite maximum ~25.25 in raw head units. No
nonfinite outputs. These are numerical comparisons, not depth-quality approval.
Keep exact mode available and do not replace the installed default.
Untested lead: the per-conv delta is one-sided (0/+1, never −1), suggesting a
systematic rounding/bias offset; a bias correction may recover exactness.

ISA gates: eight independent convolution geometries /14 input formats pass
plain and CM, with identical results and guarded blocked collectors. The
separate original-QNN aligned 1x1 leaf now passes 128 fixtures /1,966,080 output
comparisons, including CI/CO32/64/96/128, H8/16, W8/24, scattered croutons,
identity and random signed weights/bias. `check_qnn_hmx_1x1_sim.py` uses an
independent int64 oracle. Files: `qd_qnn_hmx_1x1.S`, extraction script and
`bench/qnn_hmx_1x1_sim.c`; report `qnn-hmx-1x1-sim.json`. **SIM only**, not
integrated or benchmarked on hardware. The original leaf has external branches
to unextracted variants: only the proved aligned config is admissible. Packet
operands use old register values: bias starts at the supplied pointer, not
pointer+256. Flatten spatial croutons into descriptor block count, width64.

### What blocks Model performance

Native HMX arithmetic is around 6 ms, while the working graph remains around
100 ms. Host glue, 1,046 DSP requests, transfers, and layout conversions dominate
the remaining gap. The graph uploads roughly 133 MB of packet data per run
(the counter includes logically reused weights). Full residency removes ARM
round trips but a literal DSP port over DDR makes scalar/layout accesses much
slower. Residency alone is not a performance solution.

The full resident prototype, HVX glue and private 128 MB module allocation are
implemented; see `RESIDENT_EXECUTOR_WIP.md`. Earlier smoke-first tests concealed
a cold-random failure at command 14. The current combination of shared-uncached
mutable pools, proper `dccleaninva` publication, and heap publication before
HVX metadata reads passes the previously failing cold-random cases for both
models. A fresh-nonce ownership probe verifies every 4 KB page read-only before
any writes, then verifies DSP writes and JTLB restoration. Two sessions on each
of two allocation leases pass. Attribute C7 for scalar pages showed no timing change, but
the cached-mode reports lack a bootstrap SHA, so it is unproven which mapping ran.
Cached mutable pools with only the `dccleaninva`/heap-publication fixes have not
been tested; uncached C6 pools are a likely contributor to warm time. Do not claim an isolated root cause from these combined
changes or promote older smoke-first results.

Corrected resident timings are about 1.68–1.71 seconds warm and 12.9 seconds on
first cold inference. The cold excess is **not** weight preparation
(`prepare_weights_ms` ~200 ms): ~9.2 s is inside `ConvLayer_s1.opt@CB*2.Xwb.Fi.t`
(629 calls, ~213 ms warm) and ~2.0 s inside `ConvLayer_s2.opt@CB*2.Xwb.Fi`
(80 calls, ~33 ms warm), uncovered by any phase counter; see `RESIDENT_EXECUTOR_WIP.md`.
HMX/HVX workers total about 52–55 ms. The corrected resident backend passes all 54 inputs plus 4 warm repetitions,
and separate cold-random starts for both models. Reports are
`resident-final-full-validation.json` and `resident-final-cold-validation.json`.
Module cleanup is recorded in `RESIDENT_EXECUTOR_WIP.md`; this remains an
experimental backend.

The selected route toward low latency is native kernels consuming crouton
blocks directly in VTCM, plus DSP glue in that same storage layout. The 1x1
ISA gate proves one building block. Spatial/u16 kernels, full VTCM ownership,
live-block/scratch scheduling, and end-to-end integration are outstanding.
The current verified scratch is only 0x2d0000 bytes, despite an 8 MB VTCM mapping;
do not assume the rest is available or overwrite live graph blocks.

750 MHz is the DSP AXI master clock, **not DDR**. Bounded 750/1000/750 testing
gave vector 10.09/10.33/10.15 GB/s and dependent scalar 200.8/196.1/201.6 ns;
only ~2–3% benefit from 1000 over 750. Default stays 750. No DDR-controller clock change was made.
