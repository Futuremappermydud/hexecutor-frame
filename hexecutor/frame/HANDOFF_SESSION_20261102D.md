# Handoff: 2026-11-02 session D (82.6 -> 68.1 ms) and the plan to Model parity (~10 ms)

Continues `HANDOFF_SESSION_20261102C.md`. The user's goal is Model-level latency; Model telemetry
shows ~10.5 ms disparity (~14.3 ms "core"; what that boundary includes is unconfirmed,
`NATIVE_EXECUTOR_HANDOFF.md`). Exactness policy is unchanged (`exactness-strategy` memory):
exact first, as far down as possible, then a QNN-like non-exact mode behind a compile flag.

**State: everything is committed.** HEAD is `e111bdb` on `main`.

- Full `ab.sh` gate on `e111bdb` (both models, smoke and random, repeat 2): **passed, 68.1 ms primary
  / 68.4 ms alternate** (`l2hw`). Reports are in `frame/build/ab/slgate/`.
- The Frame was rebooted twice this session and is clean: `cdsp_mem` loaded, chunk map 0 bad.
- Untracked and not mine: `depth-model/0`, `depth-model/10000` … `80000000` (empty-looking numeric
  files) and `model-assets/`. I left them alone.

---

## 1. Results this session (Frame, `l2hw`, steady ms, primary smoke)

| Commit | Change | ms |
|---|---|---:|
| `5baf580` | Session C tree, full gate run and committed (chain memo, whole-block ADD) | 82.6 |
| (in e111bdb) | ADD_U8 whole-block in 16-bit HVX math | 80.2 |
| `e111bdb` | **Staging-free 3x3 conv on silicon** (staged 3-row window); `l2hw` now includes it | **68.1** |

Every row is exact against the hw oracle. `l2hmx` (emulator-exact rounding) was 112.2 ms at `5baf580`.
The slide path only exists in hw mode.

---

## 2. Staging-free 3x3 HMX conv: working on silicon

Design and decoding are in session C, section 2. What changed:

### Silicon bisect (first eligible call, cmd 171, `SLIDE_LIMIT=1`)

| Build | Frame |
|---|---|
| In-place input, dummy stores (session C) | **hang** |
| + `barrier` and HVX read-back before HMX | **hang** |
| + `SLIDE_NODUMMY` only | **hang** |
| `SLIDE_STAGE` + `SLIDE_NODUMMY` | **pass, exact** |
| Full model, `SLIDE_STAGE` + `SLIDE_NODUMMY` | **pass, exact, 68.3 ms; 6/33 21.4 -> 9.3 ms** |

So the trigger is **reading the input blocks in place**. Rt carries `B - A` between blocks scattered
across pool 1 and a workspace zero block, which can make the offsets large and negative. Session C
ruled out all-positive deltas alone (`SLIDE_POSDELTA`). The working form copies each strip's 3 input
block rows into one contiguous workspace window, so every delta equals the small positive row
stride. That is what QNN's probe had (`native_kernels/xwb1022_frame_probe.c`: one contiguous VTCM
input region). Whether STAGE alone is enough without NODUMMY was not tested; both are on.

**Neither ISS mode catches this.** Functional mode passes everything. `--timing` mode is useless for
HMX: even the silicon-proven `qd_hmx_finish` never completes under it (control run in
`.work/slidetiming/`). **Never trust an HMX mode change that has only run in the simulator.**
Bisect on silicon with a one-call limit.

### Code (all in `e111bdb`)

- `prepared_hmx_conv.inc`:
  - `NPRG2_SLIDE3` now implies `SLIDE_STAGE` and `SLIDE_NODUMMY`. `-DSLIDE_INPLACE` restores the
    hanging form, for experiments only.
  - The staged window is `3*ncol*cbs*2048` bytes after the C blocks in the workspace. Pads are
    zero-filled, followed by a `barrier` and an HVX read-back.
  - Calls whose window doesn't fit the workspace are declined.
- `qd_hmx_ops.S` `qd_hmx_slide3_n`: a store whose Rs word is 0 is skipped. The `cvt` still runs, as
  QNN's does. `qd_hmx_ops_encoded.S` was regenerated:
  ```sh
  docker run … hexagon-clang -mv75 -mhvx -mhvx-length=128b -mhmx -c frame/qd_hmx_ops.S -o .work/enc/o.o
  python3 frame/bench/encode_hmx.py .work/enc/o.o --source frame/qd_hmx_ops.S --output frame/qd_hmx_ops_encoded.S
  ```
  `encode_hmx.py` runs on the host; the docker image has no python3.
- `bench/ab.sh`: `l2hw` adds `-DNPRG2_SLIDE3=1`. `l2hw0` is the previous build (no slide).

### Cost now and what's left in 6/33

629 calls, 9.3 ms (14.9 us/call). The staging copy is 3 block reads per column per strip. A rolling
window (keep 2 of the 3 rows from the previous strip, copy 1 new row) would cut copying 3x. Per-call
setup (Fi build, near copy, rec/st tables) is now a large share. Expect ~3–5 ms more from
both together.

---

## 3. 16-bit HVX recipe for u8 elementwise (exact)

`blocked_glue/add_u8.c` `add_vec16`. ADD is `(a*ma + b*mb + 2^(s-1)) >> s`, clamped to 255. When
`ma, mb < 2^16` and `1 <= s <= 15`:

```
unpack ub -> uh (lo = bytes 0..63, hi = 64..127)
w  = vmpy(Vuh a, Ruh ma|ma<<16)            // uh x uh -> uw pair (even/odd lanes)
w  = vmpyacc(w, Vuh b, Ruh mb|mb<<16)
h  = vasr(hi(w), lo(w), s):rnd:sat          // uw -> uh, adds 2^(s-1); re-interleaves
h  = vmin(h, 255);  out = vpack(hi_h, lo_h):sat
```

- **Pitfall:** the `vasr` shift is taken mod 16, so `s=16` is wrong. The guard is `s <= 15`; the
  model uses 13–15.
- Exhaustive check: `.work/ew16/add16_test.c`, built by `.work/ew16/run.sh` in the
  `depth-model-hmx-tools` image. It runs 400 parameter sets (edge multipliers 0/1/255/256/…/65535 and
  every shift) over all 65,536 (a,b) pairs, with 0 bad.
- Speed: **380 -> 57 cycles per output vector** in the sim (6.7x). On the Frame, 10/13 ADD went
  from 1.6 to 0.49 ms.
- This is the template for REQUANT_U8, LINEARCLIP and the u16 elementwise ops (section 6, A1).

---

## 4. Where the 68 ms goes now

`frame/build/ab/slgate/sf_primary.json`, steady run. Handler time is 61.4 ms over 5,068 commands;
1,299 are no-effect. Fixed costs: preflight 1.9, VTCM in 2.3 / out 0.2, dispatch 0.9.

The profile's variant number is a **1-based index into `resident_schema.VARIANTS`**, so 7/31 is
`VARIANTS[30]`.

| op/var | kernel | calls | ms | us/call |
|---|---|---:|---:|---:|
| 6/33 | `ConvLayer_s1.opt@CB*2.Xwb.Fi.t` (u8 3x3, slide) | 629 | 9.34 | 14.9 |
| 7/31 | `ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4` | 174 | 4.60 | 26.4 |
| 6/35 | `ConvLayer_s1.opt@CH*2.Xwb.Fi.t` (u16 conv) | 70 | 4.59 | 65.6 |
| 18/56 | `Transpose.2D@CH*2.t` | 18 | 3.35 | 186.1 |
| 6/36 | `ConvLayer_s2.opt@CB*2.Xwb.Fi` (stride 2) | 80 | 2.97 | 37.1 |
| 6/32 | `ConvLayer_s1.opt@CB*2.Xpb.Fi.t` (dynamic weights) | 217 | 2.76 | 12.7 |
| 18/55 | `Transpose.2D@CB*2.t` | 82 | 2.64 | 32.2 |
| 25/63 | `d2s.shortD.flatout@FH.CH.fi` | 10 | 1.78 | 177.7 |
| 39/68 | `linearclip@CH.CB.fi` | 40 | 1.74 | 43.6 |
| 32/21 | `conv_scale_from_weights@ff.FB` | 80 | 1.50 | 18.7 |
| 24/51 | `ResizeNear_slice.tcm_special@CB*2.fi*2` | 107 | 1.48 | 13.8 |
| 4/30 | `weights_to_vtcm@Xwb.xwb.` | 64 | 1.48 | 23.1 |
| 41/72 | `mul_op@CH*2.fH.fi` | 10 | 1.41 | 141.5 |
| 20/46 | `Gather@FB*2.fi*3` | 9 | 1.27 | 140.7 |
| 29/38 | `Dequantize@Ff.CH` | 5 | 1.23 | 246.3 |
| 23/37 | `ConvLayer.opt.convert_weights_to_signed.packed@xpb.FB.t.fs.t` | 13 | 1.10 | 84.4 |
| 14/74 | `mul_op_flat@FH*2.fH.fi` | 50 | 1.09 | 21.7 |
| 22/52 | `SlicePad_shape_inplace@FB*2.s4*3.cB` | 10 | 1.02 | 102.5 |
| 5/24 | `inconv_im2col.tcm@CH.FH.s4.t` | 80 | 1.01 | 12.6 |
| 9/71 | `mul_op@CH*2.cH.fi` | 54 | 0.96 | 17.7 |
| 39/67 | `linearclip@CH*2.fi` | 30 | 0.90 | 30.0 |
| 13/59 | `addsub_op@CH*2.fH.fi` | 60 | 0.84 | 13.9 |
| 16/49 | `Reshape.hw@CB*2` | 80 | 0.82 | 10.3 |
| 30/39 | `ElementWiseSign@CH*2` | 12 | 0.82 | 68.1 |

The full named table (52 rows) is in `.work/named_profile.md`. Rerun the snippet from this session on
any report to regenerate it.

| Bucket | ms |
|---|---:|
| Convs (5 variants, 1,036 calls; HMX MAC time is only ~5 ms of it) | 19.9 |
| **Non-conv** (~70 variants, ~2,700 effective commands) | **41.4** |
| Fixed (preflight + VTCM in/out + dispatch) | 5.3 |

---

## 5. Gap analysis: what keeps us from Model's ~10 ms

**Model runs the same schedule.** `recovered/primary/operations.csv` is QNN's post-prepare schedule,
and our 5,068 commands are that schedule minus Shape/Const/Dummy (`QNN_RESEARCH.md` section 1). QNN
did its fusion offline. So the gap is **per-op speed, not op count**: about 12 us per command for us
versus about 2.8 us on Model.

1. **Biggest blocker: the non-conv long tail, 41 ms.** That's 4x Model's whole budget.
   - No single op is above 4.6 ms, and the top 15 add up to only ~27 ms.
   - The cause is how the kernels are written:
     - exactness-first 32-bit lane math;
     - per-vector or per-element address lookups through block tables;
     - generic `t_read_region` / `anew` / `t_write_region` fallbacks (whole-tensor copy, then
       permute, then copy out) whenever a fast path declines.
   - Section 3 shows the size of the gap on one kernel: 6.7x, still exact.
   - Closing it means rewriting ~20 kernels, each with an exhaustive check.
2. **Serial execution.** Model overlaps HMX convs, HVX ops and DMA across 4 HVX threads
   (`hvx_threads=4`, `dma_prefetch_distance=4`, `df_subsequencer=por_dp`). We run one command at a
   time, with up to 3 parts in parallel inside one command. Once the kernels are fast, overlap is
   roughly the last 2x.
3. **Conv overhead.** 19.9 ms against ~5 ms of HMX MAC time. It's setup, Fi, staging and output
   passes, plus three conv types still on old paths (6/35 u16, 6/36 stride 2, 6/32 dynamic weights).
4. **Fixed per-inference costs, 5.3 ms.** That's half of Model's budget.
   - Preflight runs every inference.
   - VTCM in (2.3 ms) copies the input over the Frame's ~1.25 GB/s DDR path; Valve's DSP setup
     limits this.

**Not blockers** (checked):

| Ruled out | Why |
|---|---|
| Core clock | 1843 MHz (`HMX.md`) |
| HMX throughput | 26.8 GMAC per inference; u8 MAC phase ~5 ms |
| DDR in steady state | The engine is VTCM/L2-resident; `QDS_AXIM_RUN` gave no gain |
| Exactness | The fast ADD is exact; QNN runs the same integer kernels |

---

## 6. Roadmap to ~10 ms

Targets are estimates. "Exact" means bit-identical to the hw oracle (`--expect hw`). Gate every
item: exhaustive or random sim check of the kernel, then `ab.sh -q`, then full `ab.sh` before
committing.

### Budget

| Bucket | Now | After A–C (exact) | After D–E | Model-like |
|---|---:|---:|---:|---:|
| Non-conv | 41.4 | ~12 | ~6 | ~4 |
| Conv | 19.9 | ~9 | ~6 | ~5 |
| Fixed | 5.3 | ~4 | ~1.5 | ~1 |
| **Total** | **68** | **~25** | **~13** | **~10** |

### A. Exact HVX kernel rewrites (non-conv, largest gain): ~41 -> ~15 ms

**A1. Elementwise in 16-bit math** (section 3 recipe). Template: `qd_blocked_add_u8_apply_whole`.
Each kernel gets an exhaustive or large-random sim check against its scalar oracle in
`prepared_elementwise.c`. Copy the pattern of `.work/ew16/add16_test.c`.

| Op | Now (ms) | Notes |
|---|---:|---|
| REQUANT_U8 8/70 | 0.30 | Same form, one input |
| LINEARCLIP u8 -> u16, 39/68 | 1.74 | `blocked_glue/linearclip_u8_u16.c` |
| LINEARCLIP 39/67 | 0.90 | |
| MUL 41/72, 9/71, 14/74 | 1.41, 0.96, 1.09 | `elementwise_u16.c`; u16 x u16 needs uw products and a 64-bit-safe shift; check the multiplier ranges first, like the ADD param dump |
| addsub 13/59 | 0.84 | |
| Sign 30/39 | 0.82 | |
| Dequantize 29/38 | 1.23 | |

Estimated total ~8 ms down to ~2 ms. Before writing each one, dump its parameter ranges. Session C's
scratch `params.c` `#include`s `prepared_elementwise.c` for this.

**A2. Data movers at HVX copy speed.**
- `k_space_to_depth` (7/31, 4.6 ms) and `k_transpose` (18/55, 18/56, 6 ms) have fast paths
  (`s2d_fast`, `tr_fast_dispatch` in `prepared_datamove.inc`), but fall back to
  read-region/permute/write when those decline.
- First step: build with `-DDM_SHAPES` and turn on `DMPATH` logging in the sim to count which calls
  take the slow path, with their shapes.
- These ops are pure permutations of 2 KB crouton blocks or 32-byte lanes. Write block-to-block HVX
  kernels with `vdelta`/`vshuff`; never materialise the whole tensor.
- Next in line: D2S 25/63 (178 us/call), Gather 20/46 (141 us/call), SlicePad 22/52, Reshape.hw
  16/49, ResizeNear 24/51 (on 107 calls, 13.8 us is per-call overhead).
- Estimated 15–17 ms down to ~4 ms.

**A3. Dynamic-weight chain** (80 bmm instances, about 6 ms): 32/21, 18/55, 33/66, 34/20, 36/26,
37/22, 38/19, 4/30, 23/37. Fuse into one prep handler per bmm (`QNN_RESEARCH.md` win 3). Keep the
f32 steps in the same order and it stays exact. Estimated 6 ms down to ~1.5 ms.

### B. Conv: ~20 -> ~6 ms

**B1. 6/33 polish** (section 2): rolling 3-row window plus cheaper per-call setup. 9.3 -> ~5 ms.

**B2. Stride-2, 6/36 (2.97 ms).**
- Decode `hmx_v73_convbbb_stride2` from `.work/qnnv75/dis.txt`; it's right after stride1 at 0x344a40.
- Use the session C probe method (`frame/bench/hmx_slide/run.sh`).
- Use QNN's contiguous-window addressing from the start.
- S2D 7/31 feeds these convs; if the stride-2 leaf reads S2D's source directly, part of the 4.6 ms
  S2D goes away too.

**B3. u16 conv 6/35 (4.59 ms, 65.6 us/call)** and its im2col 5/24 (1.0 ms). Decode `hmx_convhbh_*`
(V69 legacy leaf, `native_u16/README.md`). The u16 epilogue is `cvt.uh:2x1`, already exact in
`hmx_native16_n`. Only the A addressing is new.

**B4. 6/32 dynamic-weight u8 3x3 (217 calls, 2.76 ms).** Not covered by the slide path: `hmx_slide`
is only tried when `!bcopy`, and dynamic weights take the B-copy path. Either let slide take copied
weights (the B copy goes into the workspace first), or fuse the weight prep from A3 so B lands in
hardware tap order directly.

**Rule for every new HMX mode:** first a sim-exact random test (`slide_test.c` pattern), then the
real-schedule sim gate. Then **silicon with `SLIDE_LIMIT=1`-style knobs**, expecting a reboot per
hang.

### C. Fixed costs: 5.3 -> ~1.5 ms

- **Preflight (1.9 ms) every inference.** Memoise its verdict the way `chain_run` memoises
  validators. Re-run only if the image or pools change.
- **Dispatch (0.9 ms).** 1,299 no-effect commands still cost a loop iteration each. Compact them out
  of the prepared program at load time.
- **VTCM in (2.3 ms).** Find out what is copied. If it's more than the input image (e.g. weights or
  constants restored after the lease), keep it resident or move it with DMA while the previous
  inference runs.

### D. Overlap and concurrency: ~2x on what remains

- Run HMX convs concurrently with independent HVX commands. The schedule's `@SyncOp`/DMA checkpoints
  mark dependencies; QNN's subsequencer uses them.
- A minimal version: while `hmx_slide` waits for HMX completion, workers run the next command if
  it's independent. That needs a dependency check from the command's input/output pools.
- **Try a 4th HVX thread.** Unverified whether Valve's DSP stack allows it, so check
  `QDS_PREPARED_THREADS=4` on silicon.
- Keep using `nprg2_parallel_for` for the per-command split.

### E. Non-exact QNN-like mode (`-DNPRG2_FAST`), last

Only after A–D. Candidates and error metric are in `QNN_RESEARCH.md` "Non-bit-exact specifics" and
"Error metric proposal". The rounding mode (`hw`) is already the QNN behaviour, so expect little
from this.

---

## 7. Tools, gates, recovery

**Frame gates** (run from `depth-model/`):
- `sh frame/bench/ab.sh -q -t TAG VARIANT…`: primary smoke only, ~1 min.
- `sh frame/bench/ab.sh -t TAG VARIANT…`: full gate.
- Custom variant:
  `NAME:QDS_PREPARED_THREADS=3,QDS_PREPARED_FUSE=1,QDS_PREPARED_HMX=1,QDS_L2_CACHE=1,QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1+-DNPRG2_SLIDE3=1+-DX`.
  Commas separate QDS keys; `+` separates defines.
- **A custom variant must list the whole `l2hw` set**: only `base()` is implied.

**Sim gate (real schedule, ~2 min per variant):**
```sh
CFLAGS_EXTRA="-DNPRG2_HW_ROUNDING=1 -DNPRG2_SLIDE3=1" SEGDIR=.work/seg40 SEGMENT_GLOB='primary-seg[12]' \
  sh frame/native_program/run_prepared_reference_sim.sh primary segments 40 fast hmx fuse
```

**Host gate:** `test_prepared_reference_sim <model> split 1 <dir> fast`, built with cc under both
`-DNPRG2_HW_ROUNDING=0` and `=1`, ~3 s (session B notes).

**Kernel checks:**
- `.work/ew16/` (ADD pattern)
- `.work/setupbench/{qatest,dqtest,difftest}.c`
- `frame/bench/hmx_slide/` (HMX ISS probes and `slide_test.c`)

**Slide debug defines:** `SLIDE_TRACE`, `SLIDE_LIMIT=n`, `SLIDE_ONLY_CBS=n`, `SLIDE_POSDELTA`,
`SLIDE_INPLACE` (hangs silicon). The experiment-only `EXP_EW` defines are still in
`prepared_elementwise.c`; remove them when you're done with them.

**Recovery after a CDSP hang** ("VTCM lease exit did not complete" / "CDSP request timed out"):
1. `ssh frame 'echo frame1 | sudo -S systemctl reboot'`
2. Wait for ssh, then `cd ~/frame-dsp/depth-model/wave1-20261101`.
3. `echo frame1 | sudo -S insmod frame/kmod/cdsp_mem.ko size_mb=128`
4. Run `/home/steamos/frame-dsp/depth-model/.venv/bin/python frame/bench/chunk_map.py` **as
   steamos, not via sudo**: root can't open the steamos-owned `/tmp/depth-model-cdsp.lock` (sticky
   /tmp).

Reboot recovery is authorized.

**Connectivity:** `ssh frame` (Wi-Fi). The Frame may be off the network while the user is away.
In that case do sim and host work only, and queue silicon tests.

## 8. Hazards learned (add to these)

- **The ISS accepts HMX addressing that silicon rejects** (section 2). Silicon wedges the whole CDSP:
  the engine's 1 s completion timeout never fires.
- **HVX `vasr` shift amounts are mod 16** for the uw -> uh narrowing. Guard shifts and test the edge.
- The profile variant number is a 1-based index into `resident_schema.VARIANTS`.
- After an exact-path change, the full gate (alternate model + random inputs) is the real check.
  `-q` only covers primary smoke.
