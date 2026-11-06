# Plan: DSP-resident exact inference under 100 ms (written 2026-11-01)

Target: **one primary-model inference (the models are alternatives, see Open questions), fully on the CDSP (no ARM in
the loop), bit-exact against the emulator hashes, < 100 ms steady state** on
the Frame.

## Where we are (main @ 3320a70, Frame, primary smoke, exact)

| Build | Steady ms |
|---|---:|
| merged, 3 threads (`QDS_PREPARED_THREADS=3`) | 979 |
| merged, L2 + fusion | 1054 |

Time split for the 3-thread build (handler run ms, 968 total):

| Bucket | ms | Notes |
|---|---:|---|
| u16 conv 6/35 (70) + 6/34 (40) | 390 | HVX fast conv; 6/35 barely threads (helper slot too small) |
| u8 conv 6/33 (629) + 6/36 (80) + 6/32 (217) | 289 | HVX fast conv, mostly 2 parts |
| conv phases (part 0 only) | staging 379, MAC 208 | **staging costs more than the MACs** |
| non-conv | 289 | 5/24 47, 41/72 31, 40/11 30, 29/38 29, 8/70 18, 25/63 15, 10/13 15, … |
| executor | preflight 35 (first call only, reused after), dispatch 9 | |

Facts that set the plan:
- 26.8 GMAC per inference, 96% u8 x s8 (REALTIME.md). The HVX `vrmpy` peak on
  3 threads is 366 GMAC/s, so **HVX alone has a 73 ms floor at 100% efficiency.
  The target needs HMX.**
- An exact HMX conv already exists and is proven: the hybrid's v6 path
  (`qd_hmx_kernel.h`, `qd_hmx_wide`/`qd_hmx_finish`). It puts u8 activations in
  FP16 lanes (`<<8`), so products and sums are exact, then requantizes on HVX.
  It was 54/54 bit-exact, and the whole hybrid ran at ~115 ms with ARM glue.
  Its profile: HMX calc ~10 ms, input prep ~21, quantize/copy-out ~11, copy-in
  ~8, u16 HVX convs ~13 (NATIVE_EXECUTOR_HANDOFF.md).
- The native integer QNN leaf (v7/v8) is **not** exact (0/+1 conversion
  delta). Do not use it for the exact target.
- The VTCM pool-1 mirror takes 7.5 MiB. Only the 512 KiB tail is free for conv
  workspace, HMX A/W tiles and helper slots.

## Progress (2026-11-01, late)

Frame, thr3fuse (`QDS_PREPARED_THREADS=3 QDS_PREPARED_FUSE=1`), exact on both
models, smoke + random, repeat 2, lease and guards clean. Run with
`sh frame/bench/ab.sh -t TAG thr3fuse`; reports and table in `frame/build/ab/TAG/`.

| Commit | Change | Steady ms |
|---|---|---:|
| c8b322b | Step 0: A/B harness, phase split u8/u16 over all parts (baseline thr3 946.7, l2fuse 1043.8) | 925.0 |
| dbf25b8 | Step 1: u16 direct staging with HVX (`stage_row16`): u16 staging 360 → 3 ms, 6/35+6/34 390 → 33 ms | 565.3 |
| (s2a) | Step 2: 3 parts for 8-row conv tiles (cap was oh/(kh+1) = 2) | 527.5 |
| f9d473b | Step 4: HVX u16 blocked copies in the reference (`blocked_run4_16`): non-conv 220 → 157 ms | **468.4** |

Findings that change the plan:
- Step 1's cost was the u16 path only: u16 inputs were never staged direct and
  went through `blocked_impl`'s scalar 2-byte-per-element VTCM copy. u8 staging
  is only ~40 ms (CPU, all parts).
- Step 2.1 (helper slots to VTCM) would be a regression: `tile_block` reads
  activations with scalar loads (`vsplat_R(*(uint32_t *)x)`), so the row ring
  must stay in L2-TCM. The 0x3e00 slots already fit every shape (census:
  `-DCONV_SHAPES` host build prints PARTS lines).
- Conv wall now (6/33 212 ms) ≈ setup 35 ms (serial, before the split:
  `build_taps`, `tap_wsum`, `rq_prepare`, constant per command → cacheable at
  preflight) + the 3/8 row imbalance of an 8-row tile over 3 parts.
- Remaining non-conv (157 ms) is the scalar int64/double reference compute:
  29/38 DEQUANTIZE 17, 41/72 MUL_INLINE 11, 28/48 QUANTIZE 10, 35/23 10,
  40/11 ABS 10, 7/31 10, 23/37 9, 18/56 8 ms.

Next, in order: Step 3 (HMX u8, ~200 ms of the remaining ~300 ms conv);
cache per-command conv setup; typed/HVX kernels for the non-conv list above.

## Budget at the target

| Bucket | Today | Target |
|---|---:|---:|
| u8 conv (HMX exact) | 289 | 35 |
| u16 conv (HVX, 3 threads) | 390 | 20 |
| non-conv | 289 | 25 |
| executor, VTCM in/out, lease | ~12 | 8 |
| **Total** | **979** | **~88** |

## Steps

Every step ends at the same gate: exact on both models, smoke and random,
repeat ≥ 2, lease and guards clean, plus a Frame timing table vs the previous
step. Commit after each passing step.

### Step 0 — One-command A/B harness (½ day)
1. Add `bench/ab.sh ENGINE...`. It builds each variant, copies it to the Frame,
   runs `verify_l2_short` then `verify_prepared_full_model` for both models
   (`--repeat 2`), and prints a per-family table (conv by variant, non-conv top
   15, phases, thread record).
2. Make the conv phase counters cover helper parts (sum per part, or
   wall-clock per command). Today they cover part 0 only.
3. Freeze baselines: thr3 979 ms and l2fuse 1054 ms.

### Step 1 — Remove conv staging (390 → ~40 ms staging), u16 first
Staging (`stage_row8`, `near_split16`, `near_copy` into the near row ring) is
379 ms, more than the MACs.
1. Measure staging by variant. Find whether the cost is scalar address math
   per row, VTCM→C7/TCM traffic, or redundant re-staging per column chunk.
2. Stage once per command into the VTCM tail with HVX, split across the 3
   threads, or let `tile_block` read rows straight from the blocked crouton
   input (no copy) where the alignment allows.
3. Use the u16 hi/lo split during staging (`near_split16`) for 6/35 and 6/34.
4. Target: 6/35 + 6/34 ≤ 120 ms single-threaded. Sim-check the kernel
   against the reference conv on all real 6/34–6/35 shapes.

### Step 2 — Threads at full width (u16 conv 120 → ~45 ms)
1. Move the helper per-thread slots (acc + row ring) from L2-TCM
   (`0x300f0200..`, 15.5 KiB) to the VTCM tail. They are HVX-accessed, so
   VTCM is fine. Size them to fit 6/35 (ci up to 128+).
2. Fix why follower 2 rarely runs. `want = min(1+par_slots, oh/(kh+1))`
   likely caps it at 2; split by output column chunks as well as rows.
3. Tune the elementwise/datamove thresholds on hardware (thread record: ew/dm
   jobs are 0 today).
4. Target: thread record shows conv parts ≈ 3x jobs and u16 conv ≤ 45 ms.

### Step 3 — Exact HMX for u8 convs (289 → ~35 ms) — the big step
1. Wrap the v6 exact HMX kernel (`qd_hmx_kernel.h`, `qd_hmx_wide`,
   `qd_hmx_finish`, `hmx_layout`) as a prepared handler for 6/33, 6/32 and
   6/36. Take the shape families the native binder already accepts
   (EXECUTOR.md: 132 1x1 + 583 3x3 records) as the first coverage set, and
   keep the HVX fast conv as the chained fallback.
2. Input prep: build the A tiles (u8 in FP16 lanes, ×9 expansion for 3x3)
   from the blocked VTCM activations with `vscatter` across 3 threads. Tile
   by output rows so A fits the tail. The budget is 512 KiB minus W tiles.
3. Weights: convert once in `prepared_open` to the HMX FP16 tile layout in
   private DDR (~2x the int8 size, ~14 MB). Copy each conv's tiles into VTCM
   per command with HVX or user DMA (`bench_dma.c`). Measure this copy first:
   at ~1.3 GB/s DDR it is ~11 ms per inference. Keep the L2 lease on if it
   helps the stream (step 5).
4. Requant: reuse the exact fused HVX `rq_emit` (and the fusion epilogue)
   on HMX's int32/FP16 output, writing straight into the output blocks.
5. HMX ownership/power and SSR bit 26 save/restore, as in `qd_graph_native_direct.h`.
   Respect the frame-dsp-hazards rules: drain HMX before return, and never
   dmpause a running DMA engine.
6. Gates: V75 sim per-shape exactness against the int64 oracle on every
   real u8 shape; hardware bounded ranges first (`verify_l2_short` windows over
   the first HMX commands), then the full model.
7. Fallback if the tail is too small: free VTCM with a pool-1 liveness
   compaction. The fusion plan's liveness data already tracks dead blocks.
   Alternatively split HMX work into more, smaller tiles.

### Step 4 — Non-conv tail (289 → ~25 ms)
1. 5/24 im2col (47 ms) should disappear once HMX consumes blocked input
   directly. If not, use the layout worktree's `im2col_u8_fast` on 3 threads.
2. For each of 41/72, 40/11, 29/38, 8/70, 25/63, 10/13, 28/48, 35/23, 7/31,
   23/37: name the op, then write an HVX handler (the layout/quick patterns),
   thread-split it, or fuse it into the producing conv epilogue
   (fusion covers REQUANT_U8/ADD_U8; extend it to LinearClip and u16).
3. Finish the quick worktree's leftover gates: the `run_prepared_ew_new_sim.sh`
   windows and the Hexagon compile of the sign/lc16 glue.
4. Target: every non-conv handler < 3 ms, total ≤ 25 ms.

### Step 5 — L2 cache, revisited (decide keep/drop)
Most hot data is now in VTCM. L2 matters only for the HMX weight stream and
DDR spills.
1. Switch the lease to L2CFG=1 (512 KiB cache). The lower 512 KiB stays TCM:
   tables, expanded blocks and conv near at `0x30078000` stay fast.
2. Move the follower stacks and the hot metadata into the lower half or onto
   C7 page 7 so threads and L2 coexist. Remove the build guard.
3. A/B against L2 off. Keep L2 only if it wins at the current state.

### Step 6 — Executor overhead (~12 → ≤ 8 ms)
1. Preflight reuse is in (first call 35 ms, reused after). Make sure the
   steady-state path never re-validates. The 5068 commands' dispatch costs
   9 ms, about 1.8 µs each: strip NOPs and fused-away commands from the run
   list (quick's strip bitmap).
2. The VTCM in/out copy (pool 1, 1.9 ms) and the lease (~0.6 ms) are fine as
   they are.

### Step 7 — Productize
1. Steady-state stereo runner on the resident engine: alternate models,
   measure end to end including ARM submit and readback.
2. Cold start, failure cleanup inside the lease and the thread join timeout,
   and a 1000-iteration soak with exactness checked on each output.
3. Make it the default backend only after the soak passes.

## Order and expected trajectory

| After step | Expected steady ms |
|---|---:|
| 1 staging | ~700 |
| 2 threads | ~550 |
| 3 HMX u8 | ~230 |
| 4 non-conv | ~65–90 |
| 5–6 | ~60–85 |

Steps 1–2 and step 3 touch different conv paths (u16 HVX vs u8 HMX), so they
can run in parallel worktrees. Step 4 can also run alongside, one op per
worktree.

## Risks
- **VTCM tail capacity** for the HMX A/W tiles (step 3.2/3.7): the most
  likely blocker. Prototype the tiling against the largest 3x3 (ci=128) shape
  first.
- **Weight bandwidth** for HMX FP16 tiles (step 3.3): ~11 ms at the measured
  DDR rate is acceptable. Much worse would need int8 weights in DDR expanded
  to FP16 in VTCM by HVX.
- **The 3x3 HMX stall** with scattered A pages seen earlier (handoff
  2026-10-30): compact A into the tail, which is the plan anyway.
- **Exactness** is guaranteed only for the v6 FP16-lane path. Never mix in
  v7/v8.

## Open questions
1. ~~Per inference or per stereo frame?~~ **Per inference.** The two models are
   alternatives, not a pair. The Model selects one (`Primary MLD model
   selected`, `useMlDepthType=`), and `mrsystemservice` holds only
   `stereo_model.ptl` (primary) open. Each model takes the stereo pair itself.
   Optimize the primary; keep the alternate as a correctness check only.
2. Is bit-exact against the emulator a hard requirement? If a ±1 LSB budget
   is acceptable, the native v7/v8 HMX leaf saves ~15 ms over exact.
