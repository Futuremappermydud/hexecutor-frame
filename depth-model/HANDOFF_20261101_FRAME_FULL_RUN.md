# Handoff 2026-11-01: from reference executor to a full-model Frame run

Nothing in this work is committed. Do not commit unless the user asks.

Read first, in this order:

1. [Reference-executor handoff](HANDOFF_20261101_REFERENCE_EXECUTOR.md): what the reference
   kernel is and what has been proven.
2. [Frame master handoff](frame/HANDOFF_20261030_2145.md): device rules, memory map,
   frozen-source build hygiene, and the accepted prepared engine (`74d31b4b...`).
   Its rules on hardware access, bootstrap choice, and recovery still apply.

## Goal

Get one **full-model, exact** run of both prepared models on the Frame through
the NPRG v2 prepared executor:

- all 5068 primary and 5040 alternate commands in one prepared call;
- both graph outputs equal to the emulator smoke sha256 (the same check the
  host tests use);
- full VTCM lease restoration and a clean device afterwards.

The reference kernel makes this possible today, but slowly. The work is to:

1. wire the reference kernel into Frame;
2. make convolution fast enough that the run is practical and the timing is
   meaningful;
3. prove each step offline before any hardware run.

The 20 ms target is not in scope for this handoff.

## Where things stand (2026-11-01, 14:50 EST)

- **Host:** every command of both models runs through the executor, in
  reference-only mode and in mixed mode. Lockstep matches `qd_graph` CPU after
  every command, and the whole-schedule run produces the emulator smoke output
  sha256.
- **Scratch:** the reference scratch peak is 1,647,936 bytes.
- **Pool-1 ceiling:** no command writes above 0x780000.
- **V75 simulator:** captured CPU prefixes (primary commands 0–131, alternate
  0–99) match bit for bit.
- **Full-schedule simulator run with the fast conv path: PASS** (2026-11-01,
  15:16–15:30). All 12 segments (6 per model) match all seven pools and both
  outputs against the host end state, and the final outputs equal the
  emulator smoke sha256. See "Progress" below. The pure scalar-reference
  simulator run was stopped and is not proven end to end.
- **Frame: full-model exact run PASS on hardware** (2026-11-01, 15:41–15:46),
  both models, smoke and random, about 15.5 s per inference. See "Hardware
  results" below.
- **OpenXR app:** `frame-spatial-runtime` selects this engine with
  `--depth-model-prepared`. On a captured pair its depth is byte-identical to
  the v5 hybrid backend. The one app session left `dsp_service` stuck at
  control `(2,0,8)` on shutdown, and only a reboot recovered it. That shutdown
  problem is open.

## Hardware results (2026-11-01, 16:05 EST)

**Engine:** frozen tree
`.work/frame-full-run-20261101/native-prepared-reference-first-source/`
(`BUILD_FLAGS.json`, `source-sha256.json`, `PROVENANCE.json`, `sha256.json`).
Flags: the accepted 74d set with `VECTORIZE=0`, `PREPARED_ELEMENTWISE=0`,
`PREPARED_PREPROCESS=0`, `PREPARED_COMPACT_3X3=0` and `PREPARED_REFERENCE=1`.
The on-device registration is therefore fast conv + weights + core-safe ops +
reference, a subset of the host `mixedfast` configuration.

- Engine `qd_graph_dsp-scalar-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-1x1-native-3x3-direct-range-prepared-reference-vtcm-lease.elf`,
  SHA `c432a7140ac633d9e20366ac8d801de1b42acdd2bf21dc2fdde65d415e9ce13b`, text about 112 KiB.
- The frozen bootstrap rebuild is byte-identical to the accepted guarded
  bootstrap `628e362d...`. The shared `frame/build` bootstrap was not written.
- The offline gate was rerun first: host lockstep and whole-schedule passed for
  both models in `fast` and `mixedfast`. The fast-conv V75 segment run postdates
  the last `prepared_reference.c` edit.

Results, all under `.work/frame-full-run-20261101/reports/`:

| Run | Result | DSP time |
| --- | --- | ---: |
| primary `[0,132)`, smoke + random | pools 1/3/4/5/6 exact vs CPU | 322 ms |
| alternate `[0,100)`, smoke + random | pools exact | 167 ms |
| primary full (5068), smoke + random, ×2 each | both outputs = emulator sha256 | 15.50 s |
| alternate full (5040), smoke + random, ×2 each | both outputs = emulator sha256 | 15.52 s |

- Each full case runs once from zeroed mutable pools, then again **without** a
  pool reset; both pass. New verifier: `frame/bench/verify_prepared_full_model.py`.
- Every session: exit errors `[0,0,0]`, lease phase 2/error 0, host fingerprint
  equal, final device state clean.
- Native 1x1/3x3 calls are 0. The trace shows 5068 of 5068 validated and executed.
- DSP heap peak 4,545,536 B. The boundary copies are 2.0 ms in and 0.2 ms out.
- Per-inference time is about 135× the exact hybrid (~115 ms) and about 24× the
  inexact resident graph. Conv is the next target (Step 2 above).

**App session and the open shutdown problem.** `test_model_depth_pipeline` ran
three frames of the Oct 29 IMX616 pair through the prepared backend. Inference
was 15.4–15.9 s per frame with 16,901 valid pixels. The depth file equals the
hybrid backend's output and the Oct 29 capture (sha256 `8f2dbb6e...`). On
close, `FrameTransport.stop_persistent()` saw all three EXIT markers but control
never left `(2,0,8)`. After 60 s the C++ side sent SIGTERM and the transport
retained the session.

- The retained capture (`reports/retained-app-shutdown-report.json`) showed the
  DSP side clean: lease phase 2/error 0, JTLB clear, clock and HMX restore
  markers clear, backup stable, host fingerprint equal, last result status 0.
- `frame/bench/recover_resident_exit.py` (new, guarded) acknowledged
  `(2,0,8)` → `(2,0,1)`. The documented extra dispatch then left `(2,1,4)`; the
  loader and payload were never re-entered.
- The user approved a reboot at 16:00. State after it: `[1,0,1]`, regions zero,
  eye client running, `cdsp_mem` unloaded.
- The cause is unknown. The Gazenet and "CPU to read the results" journal
  warnings also appear in the sessions that closed cleanly. Differences from
  the verifier: the bridge pins itself to CPU 7, the session ran 3 inferences
  (verifiers ran 2 or 4), and it was driven over the app's socket.
- Reproduce with the verifier first (`--repeat 3`, then under `taskset -c 7`)
  before another app session. Patching the transport to acknowledge `(2,0,8)`
  would not have helped, so `frame_transport.py` is unchanged.

## Progress (2026-11-01, 15:30 EST)

**Step 2 (fast, exact conv), done for every conv variant.** `prepared_reference.c`
has a shape-only fast path in `k_hmx_conv` (`fast_conv`), enabled with
`nprg2_executor_register_reference_conv` and a workspace
(`Nprg2ReferenceConv`):

- It reads raw weight tiles in place (the Xpb, Xwb, Xwb-reversed and stride-2
  regrouped layouts), with no unpack. Accumulation is int32, with a per-command
  bound check (`kh*kw*ci*255*128 < 2^31`); u16 inputs split into lo/hi bytes.
  `requant` is unchanged.
- HVX (`Q6_Vw_vrmpyacc_VwVubVb`) under `__HVX__`, 128 B; portable C otherwise.
  Blocks are 8 pixels × 2 output-channel groups, with 16 register accumulators.
- Coverage: 1036 of 1036 convs per model. Workspace peak 467,456 bytes; the
  call-scratch peak dropped to 983,168 bytes.
- Host: all 8 configurations pass lockstep and whole-schedule (both models ×
  `reference|mixed|fast|mixedfast` in `test_prepared_full_model`).
- V75 simulator: prefixes and the full 12-segment run pass.

**Simulator cycles, fast path, full schedule** (`cycles` flag; handler pcycles,
validate + run). Both models are nearly identical: **about 6.5e9 pcycles per
model**, so about 6.5 s at 1 GHz on a single thread, if the simulator is
memory-accurate (it is not).

| Share | pcycles | Commands | Opcode / variant |
| ---: | ---: | ---: | --- |
| 49.4% | 3.20e9 | 629 | 6 / 33 (3x3 u8) |
| 10.1% | 0.66e9 | 217 | 6 / 32 (1x1) |
| 6.5% | 0.42e9 | 70 | 6 / 35 (u16) |
| 6.3% | 0.41e9 | 80 | 6 / 36 (stride 2) |
| 3.2% | 0.20e9 | 170 | 10 / 13 |
| 3.0% | 0.19e9 | 160 | 8 / 70 |
| about 1.9% each | | | 30/39, 39/68, 5/24, 7/31, 18/56, 14/74 |

Conv is still 73%. At about 2.7e10 MACs per model, variant 33 runs far below
the vrmpy peak. Next candidates: the scalar per-pixel `requant`, the input
copy (`xfer` from uncached DDR pools), the partial-block per-pixel fallback,
and software pipelining or wider blocks in `tile_block`. After conv, the
scalar reference kernels for 10/13 and 8/70 come next.

**Step 1 (Frame integration), built offline but not run.**

- `QDS_PREPARED_REFERENCE=1` in `build_resident_dsp.sh` requires
  NATIVE_PREPARED=1, ELEMENTWISE=0, COMPACT_3X3=0 and VECTORIZE=0. It adds the
  `-reference` engine suffix and `prepared_reference.c`.
- In `qd_graph_prepared.h`, the conv workspace is the VTCM tail at
  +0x2000, the reference scratch is 1 MiB `malloc`'d from the cached heap, and
  native conv is not registered (exact configuration). Reference chains go over
  the specialized handlers, and `register_reference_missing` covers the rest.
  `trace[15]` marks the build.
- Compiled and linked to /tmp with the accepted prepared-engine flags plus
  REFERENCE=1 (text 107,804 bytes). `build_resident_dsp.sh` was **not** run,
  because it rebuilds the shared preserving bootstrap.
- Open risks for hardware:
  - the host transport timeout is 120 s per command;
  - the 1 MiB heap `malloc` must succeed;
  - DDR pools 3, 4 and 6 are uncached for scalar kernels;
  - HVX stores to VTCM followed by scalar reads of the accumulator tile are
    believed safe from earlier hardware mixed runs, but are unproven for this
    path.

## Where the time goes

Host profile with the reference kernel on every key (`test_prepared_reference_sim both profile`);
both models are nearly identical:

| Share | Host s | Commands | Opcode / variant |
| ---: | ---: | ---: | --- |
| 80.2% | 2.33 | 629 | 6 / 33 `ConvLayer_s1.opt@CB*2.Xwb.Fi.t` (u8, stride 1) |
| 10.3% | 0.30 | 80 | 6 / 36 `ConvLayer_s2.opt@CB*2.Xwb.Fi` (stride 2) |
| 5.9% | 0.17 | 70 | 6 / 35 `ConvLayer_s1.opt@CH*2.Xpb.Fi.t` (u16) |
| 1.7% | 0.05 | 217 | 6 / 32 (1x1) |
| 0.2% | 0.005 | 40 | 6 / 34 |
| under 0.2% each | | the remaining 4,000+ | everything else |

**Convolution is about 98% of reference time.** Everything else together is
about 2%: roughly 60 ms on the host. Leave non-conv keys on the reference
kernel until conv is fixed.

On the DSP the scalar reference conv will be far slower than 2.9 s. Expect
seconds to tens of seconds per frame; this has not been measured. The simulator's
wall-clock time is not a device estimate. Use `hexagon_sim_read_pcycles()`
from `hexagon_sim_timer.h` (in the SDK's `target/hexagon/include`) to measure
cycles per command. Simulator cycles are a reasonable compute-bound estimate,
but not a memory-exact one.

## Work plan

### Step 1: Frame integration behind an opt-in flag (required for any full run)

Edit `frame/qd_graph_prepared.h` under a new flag, for example
`QDS_PREPARED_REFERENCE=1`. Keep the default build unchanged.

- **Registration order in `prepared_open`:**
  1. the existing specialized/native handlers;
  2. `nprg2_executor_chain_reference` for each registered key;
  3. `nprg2_executor_register_reference_missing` for the rest.

  `register_handlers` in `frame/native_program/test_prepared_full_model.c`
  has the exact loop; it skips NOP and non-schema keys.
- **Chain storage:** put `Nprg2ReferenceChain chains[64]` in `QdPreparedState`;
  it must outlive the executor. `NPRG2_EXEC_REGISTRY_CAPACITY` is now 128, so
  `QdPreparedState` (allocated with `calloc`) has grown; check heap headroom.
- **Scratch:**
  - `prepared_run` passes L2 0x30078000 with 0x8000 bytes, far too small.
    Pass a DDR region of at least the census peak (1,647,936 bytes) plus margin.
  - It must **not** be VTCM 0x31000000; that is pool 1.
  - Candidate: the old `qd_graph` DSP scratch at 0xa4000000 (16 MiB). It ends
    at 0xa5000000, before the lease backup at 0xa6000000. Verify it is mapped
    and unused in prepared mode before relying on it.
  - The executor passes one scratch pointer to every handler. Elementwise
    handlers (about 30 KiB of demand) would then also use DDR scratch; they work
    there, just more slowly.
  - Compute the size the way the census does, with
    `nprg2_exec_reference_scratch` over the range. Alternatively use a fixed
    2 MiB and check that preflight still passes.
- **Pool-1 limit:** `pool1_writable_bytes` stays `QD_DIRECT_ACTIVE_BYTES`
  (7.5 MiB). The census shows no reference write above it.
- **Exactness decision:** the native 1x1/3x3 HMX conv handlers are validated
  only to output differences of 0/+1 against the CPU reference (master
  handoff). If they are registered, the run is **not** exact and will not match
  the emulator sha256. Use two configurations:
  - *exact*: native conv off. Variants 32 and 33 go to the reference kernel,
    or to the Step 2 optimized kernel.
  - *fast*: native conv on, with the reference chained under it. Report this
    one as inexact.
- **Preflight:** it covers the whole range before the boundary copy, then
  `nprg2_runner_execute` validates again. With the reference kernel, both
  passes run every command's dry pass. That's cheap, but measure it.
- **Offline gate before hardware:**
  - host full-model lockstep for both models in reference and mixed modes
    (commands in the reference-executor handoff);
  - the V75 segmented simulator run all PASS;
  - a host build of the Frame registration path, where possible. Native conv
    and HVX elementwise code are not host-runnable; use the existing stubs as
    the other prepared tests do.

### Step 2: make convolution fast and still bit-exact

Work in this order. After each step, rerun host lockstep, then simulator
segments.

1. **Portable C, verifiable on the host.** Add a new handler for op 6 variant 33
   (then 36, 35, 32), chained over the reference so anything it rejects falls
   back. Ideas, cheapest first:
   - **Integer arithmetic is order-independent**, so any loop order and blocking
     is exact as long as nothing overflows. The reference accumulates u8 taps in
     int32 per (ky, kx) and in int64 across taps. Prove that int32 suffices for
     the whole sum (|w| ≤ 128, x ≤ 255, so 9·ci·255·128 must stay under 2^31:
     ci ≤ 7,300 for 3x3), and check it per command in `validate`.
   - **Weights are unpacked and re-summed on every call** (`conv_weights`: unpack,
     optional Xwb width reversal, `wsum`). Cache them per command, or precompute
     at open. W is read through the current tensor location: confirm whether
     it sits in pool 5 (immutable) or pool 1 (written by `WEIGHTS_TO_VTCM`)
     before caching.
   - Read the input region directly instead of copying it into a dense
     temporary (`t_read_region`), and write output tiles directly. Mind the
     blocked crouton layout in `blocked_impl`.
   - Keep `requant` exactly as written (`M`, `pre`, `post`, rounding, clip).
2. **HVX, for the simulator and device.** u8×i8 dot products into int32 lanes
   (`vrmpy` family) map directly onto the int32 bound above. Validate in the
   V75 simulator, which supports HVX, with the segment harness:
   - register the new handler chained over the reference in
     `test_prepared_reference_sim.c` (add a flag; `env_open` registers
     reference-only today);
   - run only the segments that contain the variant;
   - compare cycles with `hexagon_sim_read_pcycles()`.
3. **Threads, optional and later.** The Frame has worker threads, but they
   belong to the native 1x1 lease and HMX ownership checks. Don't share them
   until single-thread HVX is proven.

Native HMX for variants 33, 36, and 35 is a separate, much harder lane with
known blockers: 0/+1 inexactness, 3x3 A-page staging, S2 V73 capture, and
CH/u16 packing (see the master handoff). Don't make the first full exact run
depend on it.

### Step 3: hardware run (coordinator only)

Follow the master handoff exactly:

- frozen-source build with its flags recorded and SHA manifest written first;
- guarded preserving bootstrap;
- read device state before the session;
- one hardware owner.

Then:

1. **Bounded first:** run the reference-backed engine on primary `[0,132)` and
   alternate `[0,100)`. Compare all pools against the `.mpfx` captures; these
   ranges already pass in the simulator.
2. **Then full range,** exact configuration, smoke inputs. Check both output
   sha256 values, lease restoration, and clean state.
3. **Then measure.** Per-phase time from the trace fields, and the warm median.
   Compare with the exact hybrid (about 115 ms) and the inexact resident graph
   (647–652 ms).

## Pitfalls

- The reference must be compiled with
  `-ffp-contract=off -fno-fast-math -fno-strict-aliasing` on both the host and
  Hexagon, or Quantize/InvScale stop matching bit for bit.
- **Validation reads pool-5 constants.** Gather axis, Transpose perm and
  DepthToSpace factors are read in the dry pass. Load the constant pool before
  any census or preflight, or validation rejects (seen at primary cmd 996).
- The simulator heap is small. A segment holds a 38 MB fixture and about 18 MB
  of pools, and one extra full-pool copy made `malloc` fail. Compare in place.
- In zsh, `rm dir/*` with no matches aborts the command chain, and `$VAR` does
  not word-split (use `${=VAR}`).
- In reference-only lockstep, `test_prepared_full_model` caps each command's
  scratch at exactly its measured need. Keep that check when adding handlers:
  a handler whose run needs more scratch than its validate reported must fail
  here, not on the device.

## Commands (from `depth-model`)

```sh
N=frame/native_program
# Host: prefix, full, profile, split and replay segments
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror -I$N \
  $N/test_prepared_reference_sim.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c -lm -o .work/full/test_refsim
.work/full/test_refsim both prefix
.work/full/test_refsim both full
.work/full/test_refsim both profile
.work/full/test_refsim primary split 6 .work/full/segments
.work/full/test_refsim segment .work/full/segments/primary-seg0.rseg 50

# V75 simulator
sh $N/run_prepared_reference_sim.sh both prefix      # about 4 minutes
sh $N/run_prepared_reference_sim.sh both segments 6  # 12 parallel simulators, full schedule
# Fast HVX conv plus per-(op,var) pcycle tables; about 14 minutes
SEGDIR=.work/full/segments-fast sh $N/run_prepared_reference_sim.sh both segments 6 fast cycles

# Host full-model lockstep and whole schedule: see HANDOFF_20261101_REFERENCE_EXECUTOR.md
```

Key files:

- `frame/native_program/prepared_reference.c`: conv is `conv_weights`,
  `cpu_accumulate`, `requant` and `k_hmx_conv`, near line 1190.
- `frame/native_program/test_prepared_reference_sim.c`: modes prefix, full,
  split, segment and profile.
- `frame/native_program/run_prepared_reference_sim.sh`
- `frame/qd_graph_prepared.h`: `prepared_open` and `prepared_run`.
- `frame/native_program/prepared_native.c`: native conv registers variants 32
  and 33.
- `frame/native_program/prepared_conv_s2.c` (variant 36) and
  `prepared_conv_u16.c` (variants 34 and 35): existing specialized conv
  handlers, bounded simulator evidence only (see `STATUS.md`).
