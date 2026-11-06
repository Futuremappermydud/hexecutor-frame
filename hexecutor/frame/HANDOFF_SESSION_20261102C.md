# Handoff: 2026-11-02 session C (87.6 -> ~82 ms, staging-free HMX conv decoded)

Continues HANDOFF_SESSION_20261102B.md. Goal from the user: **sub-40 ms**; exact first,
non-bit-exact QNN-like mode allowed behind a compile flag (exactness-strategy memory).

**State: nothing from this session is committed.** The working tree has 13 modified
files plus `frame/bench/hmx_slide/` (new). Every change listed below is exact on the
Frame quick gate (primary smoke) and on the host gates (both models, both
`NPRG2_HW_ROUNDING` settings). The full `ab.sh` gate (alternate + random) has **not**
been run on this tree. Run it, then commit.

## Results this session (`ab.sh -q`, l2hw, steady ms)

| Change | ms |
|---|---:|
| start (92c7b57) | 87.6 |
| chain_run memoises its wrapped validator's verdict per command | 85.2 |
| ADD_U8/REQUANT_U8 whole-block HVX path | 83.9 |
| fresh reboot (same code, `-DNO_SLIDE3`) | 82.1–82.4 |
| `QDS_AXIM_RUN=1` (AXI-master clock kept at 750 MHz during the run) | 82.3 (no gain) |

Profile of the current build: `frame/build/ab/s3axim/` (`base` variant).

## 1. Shipped (exact, in the working tree)

- **`prepared_reference.c` `chain_run`.** The elementwise chain re-ran its full validator
  every run, including `alias_proof` with a heap sort of every block address. It now caches
  the verdict in `Memo.chain[cmd]`. The validators read only geometry, call records,
  pool bases and the constant pool (pool 5), so the verdict is static.
- **`blocked_glue/add_u8.c` `qd_blocked_add_u8_apply_whole`**: one block-table lookup
  per 2 KB block, 16 aligned vectors. Eligible when all regions are whole 8x8x32
  blocks. Wired in `prepared_elementwise.c` `add_apply_parts`, with host stubs.
  `mul_u32` skips the high multiply when m < 65536.
- **Measured:** with ADD's work disabled (`-DEXP_EW=1`), a command costs only
  0.6 us. **The elementwise kernels are compute-bound in their 32-bit math, not per-command
  overhead.** ADD serial is ~117 cycles per output vector.
  - Every ADD/REQUANT multiplier in the model is < 2^15, with shift 13–15 (host tool:
    scratch `params.c`, see "Tools").
  - So `vmpy` uh×uh→uw + `vasr :rnd:sat` (u16 math) would be ~3x faster, and still exact
    with an exhaustive sim check.
  - `EXP_EW` defines (1 = no work, 2 = serial) are still in `prepared_elementwise.c`.
    Experiment-only; remove them when done.

## 2. The big lever: staging-free 3x3 conv (decoded, sim-exact, hangs on silicon)

Conv 6/33 costs 21.4 ms (629 calls), of which ~14 ms is building 9x im2col A tiles.
QNN does no staging at all.

**Source.** I disassembled `libQnnHtpV75Skel.so` from
`~/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/`, function
`hmx_v73_convbbb_stride1`. Full disassembly: `.work/qnnv75/dis.txt`. I then decoded
the undocumented HMX modes on the V75 ISS (probes in `frame/bench/hmx_slide/`,
run with `frame/bench/hmx_slide/run.sh <src.c> a,b,c ...`).

**Address alignment** (`hmx_align_sim.c`). A `:deep` activation start address must be
2048-aligned: bits 7–10 are dropped, and in plain mode bits 0–6 rotate within a 128 B line.

**Non-deep activation load `activation.ub = mxmem(Rs,Rt):cm`** (`hmx_win_sim.c`):
- Rs = block A | (y<<8) | (x<<5). Rt = (B − A) | low bits.
- One load feeds T consecutive weight K-tiles. HMX generates the taps itself:
  - tap rows = y+1, starting at row 0 of A, overflowing into B (the block below);
  - (x+1) horizontal taps per row, tap order ky-major.
- Rt low bits 0x71f (x field 0): tap kx' shifts right, so out[x] = in[x−kx']. Zero fill
  at the left edge.
- Rt low 0x0ff: shifts left, with B as the right neighbour. 0x7ff: linear wrap.
  0xdf: step 2.
- `:above:cm`: Rs y field = start row s in A, continuing into B. Gives tap rows s..7.

**Sliding accumulator** (`hmx_slide_sim.c`). In right-shift mode, pixels shifted past the
tile edge land in the *next* tile. Each `cvt.ub = acc` emits one tile and slides the
window. So you stream input block columns left to right, with no `mxclracc` in between.

**Store `mxmem(Rs,Rt):cm = cvt`** (`hmx_store_sim.c`):
- QNN form: Rs = O|0xe0, Rt = 0x700. Writes block O = previous tile's pixels 1..7 into
  cols 0..6, plus the current tile's pixel 0 into col 7.
- Rt = delta|0xe0 is a masked two-block split store. It works on the ISS only; I dropped it.

**The recipe, exactly QNN's cfg words** (input view origin p ≡ 7 mod 8, i.e. the window
starts at row/col 7 of a block). Per output strip `by`, per N block `j`, per input
block column c = 0..bw+1, per cb:
- `:above:cm` with Rs = blk(by,c)|0x740, Rt = (blk(by+1,c)−A)|0x71f.
  Gives ky=0, three taps; weights 3 tiles (Rt 3071).
- `:cm` with Rs = blk(by+1,c)|0x140, Rt = (blk(by+2,c)−B)|0x71f.
  Gives ky=1,2; weights 6 tiles (Rt 6143).
- After each column: `cvt.ub`, then store to output block c−2 (dummy for c<2) with
  Rs|0xe0, Rt 0x700.
- The reversed unpacked 3x3 weight order (rev=1, all 6/33) is already the hardware tap
  order: tile cb*9 + ky*3 + kx'. B needs no copy.
- Fi is the native path's, so it is exact against the hw oracle.

**Code** (all gated, **off by default**):
- `qd_hmx_ops.S` `qd_hmx_slide3_n`, plus the regenerated `qd_hmx_ops_encoded.S`
  (via `bench/encode_hmx.py`).
- `Nprg2HmxOps.slide3_n` in `prepared_reference.h` / `qd_graph_prepared.h`.
- `prepared_hmx_conv.inc` `hmx_slide()`, called from `hmx_native` only with
  **`-DNPRG2_SLIDE3=1`**.
- Debug defines: `SLIDE_TRACE`, `SLIDE_LIMIT=n`, `SLIDE_ONLY_CBS=n`, `SLIDE_POSDELTA`.
  There's also `Memo.slide`, an input-block 2048-alignment memo.

**Validation:**
- `bench/hmx_slide/slide_test.c`: random cbs 1–8, NB 1–2, nbw up to 32. 0 bad vs the
  scalar silicon-rounding model.
- Real schedule in the sim:
  `CFLAGS_EXTRA="-DNPRG2_HW_ROUNDING=1 -DNPRG2_SLIDE3=1" SEGDIR=.work/seg40 SEGMENT_GLOB='primary-seg[12]' sh frame/native_program/run_prepared_reference_sim.sh primary segments 40 fast hmx fuse`
  gives PASS (18 slide calls, fused/staged included). `-DSLIDE_TRACE` confirms the
  calls ran.

**On the Frame it hangs the CDSP**, even with `SLIDE_LIMIT=1`: the first eligible call
(cmd 171, cbs=1, ow=256, NB=1). Recovery: reboot, `insmod cdsp_mem.ko size_mb=128`,
`bench/chunk_map.py`.

What has been ruled out or tried:
- **Ruled out: DDR.** All 6/33 tensors are in pool 1 (VTCM).
- **Ruled out: memo growth.** The `-DNO_SLIDE3` build is fine.
- **Didn't fix it: QNN-form store.** The first hang used the masked split store; switching
  to QNN's still hangs.
- **Didn't fix it: positive deltas only.** Many commands have the block below at a *lower*
  address (negative Rt delta). `SLIDE_POSDELTA` (only commands with all deltas ≥ 0) still
  hung. The `+LIMIT=1` pass earlier probably ran no slide call.
- **Untested: a missing barrier.** The path wrote the Fi table and the zero block with
  HVX stores and started HMX with no `barrier` and no HVX read-back. The old path always
  has one. `native_kernels/xwb1022_frame_probe.md` records that an uninitialised Fi made
  HMX never complete on silicon. I added `barrier` plus an HVX VTCM read-back after those
  stores. **This is the next thing to test.**

Next steps for the slide path, in order:
1. On the Frame, test the barrier fix with `-DNPRG2_SLIDE3=1+-DSLIDE_LIMIT=1`. If it
   passes, try the full model, then `SLIDE_POSDELTA` off.
2. If it still hangs, diff against the **QNN leaf that ran on silicon**
   (`native_kernels/xwb1022_frame_probe*`, `qd_qnn_hmx_stride1_generic.S`).
   - Run my `qd_hmx_slide3_n` inside that probe's harness, on its staged contiguous
     input, to separate kernel problems from environment problems.
   - Remaining candidates:
     - the dummy/zero blocks in the workspace tail;
     - packet shapes (QNN pads HMX packets with `nop`s, e.g. around `bias=mxmem2`);
     - weights 1024- but not 2048-aligned;
     - QNN never storing to a dummy (it skips the first stores via r28/r31).
3. Expected gain: 6/33 drops from 21 ms to roughly the MAC time (~2–4 ms), with no
   threads needed. The same scheme covers 6/36 (stride 2: `hmx_v73_convbbb_stride2`
   uses the same loads, decode it the same way), 6/32 and the u16 convs
   (`hmx_v75_convhbh_*`). Together these are ~30 ms of the 82.

## 3. Clocks / interconnect (user question; answered)

- The interconnect summary shows the CDSP voting avg 0 / peak 1 GB/s on
  `qnm_nsp → gemnoc → llcc → ebi`. An **earlier session already raised it**
  (`kmod/cdsp_bw.c`, `cdsp_noc.c`, `cdsp_nocclk.c`) with no effect. The DDR limit was
  the **Q6 AXI-master clock**, which Valve leaves on XO (19.2 MHz); QuRT runs it at
  750/1000 MHz (`qd_axim.h`, `llm/README.md`).
- The depth bootstrap raised it only around the VTCM save and restore.
  `QDS_AXIM_RUN=1` (`qd_graph_boot.c`) keeps it raised through the run. **No gain**
  (82.4 → 82.3): the engine is VTCM/L2-resident. I did not read the clock back mid-run;
  a cheap check would be to report `0x32008258` from the engine.
- `build_resident_dsp.sh` now passes `QDS_EXTRA_CFLAGS` to the bootstrap compile too.
- sudo on the Frame: password `frame1`; reboot recovery is authorized.

## 4. Remaining profile (82 ms) and ideas ranked

Top costs: conv 6/33 21.4 · 6/35 4.8 · S2D 7/31 4.7 · TRANSPOSE 18/56 3.4 + 18/55 2.7 ·
6/36 2.9 · 6/32 2.7 · LINEARCLIP 39/68 1.8 · ADD 1.6 · D2S 25/63 1.8 · 41/72 1.4 ·
20/46 1.3 · REQUANT 1.2 · fixed ~7 (VTCM in/out 2.1–2.5, preflight 2, dispatch 0.9).

1. **Slide conv** (section 2): ~15–25 ms.
2. **Elementwise math:** every handler in `blocked_glue/*` uses 32-bit lane math with
   per-vector address computation. Rewrite the hot ones (ADD, REQUANT, LINEARCLIP 39/68
   u8→u16, MUL_U16 14/74, REQUANT_U16 9/71, 13/59) as whole-block 16-bit HVX. Each needs
   an exhaustive sim check against the oracle (pattern: `dqtest.c`/`qatest.c` in
   `.work/setupbench/`). Estimated ~4–6 ms.
3. **Data movers:** S2D 7/31 (27 us/call), transposes 18/56 (190 us/call),
   D2S 25/63 (180 us/call), 20/46 GATHER (140 us/call), 41/72 (140 us/call). Check
   `prepared_datamove.inc` for per-element scalar paths; most are pure permutations of
   2 KB blocks and should run at HVX copy speed.
4. **Dynamic-weight chain** (80×: 18/55, 32/21, 33/66, 34/20, 36/26, 37/22, 38/19,
   4/30): ~6 ms in total. Fuse it into one handler (session B notes).
5. **Non-exact mode** (`NPRG2_FAST`-style flag), only after the exact work: see
   `QNN_RESEARCH.md`.

## Tools added this session

- `frame/bench/hmx_slide/*`: ISS probes plus a randomized slide-kernel test.
  `run.sh` builds with SDK clang (`-mhmx -lhexagon`) and runs `hexagon-sim`;
  arguments are comma-separated.
- Host helpers (session scratchpad, not in the repo; easy to recreate):
  - `shapes.c` variant printing per-tensor block-pool histograms and Rt-delta signs;
  - `params.c`, which `#include`s `prepared_elementwise.c` to dump ADD/REQUANT params.
- Gates are as in session B. Quick: `sh frame/bench/ab.sh -q -t TAG VARIANT`, ~1 min.
  Variant flags go in `NAME:QDS_...=..,QDS_EXTRA_CFLAGS=-DA+-DB`.
- After any Frame reboot, run `insmod frame/kmod/cdsp_mem.ko size_mb=128` and
  `bench/chunk_map.py` (from `~/frame-dsp/depth-model/wave1-20261101`) before `ab.sh`.

## Connectivity

`ssh frame` goes over Wi-Fi (10.20.41.156). The Frame has a USB NCM+adb gadget
(`g1`, `usb0` 10.86.200.233/29), but the UDC reported "not attached" and the Mac saw no
USB device. Over USB, use `AB_REMOTE=steamos@10.86.200.233`.
