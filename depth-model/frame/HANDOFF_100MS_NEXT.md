# Handoff: toward Model-class latency (written 2026-11-01, late)

Goal: one primary-model inference fully on the Frame CDSP, **bit-exact** against
the emulator hashes, well under 100 ms (Model's telemetry shows ~10.5 ms for the
disparity stage). The reference hashes come from the real QNN kernels in the
QAIRT 2.50 x86 HTP simulator, so QNN itself is exact while running HMX at native
rate. Our remaining overhead is software. Read `PLAN_100MS.md` (Progress
section) first.

## State on main

| Commit | Steady ms (thr3fuse+HMX, exact both models) |
|---|---:|
| f8a9986 (HEAD) | 341.1 |
| uncommitted (see below) | 327.8 (primary smoke only) |

**First action:** the working tree has two uncommitted, validated-on-host changes:
`qd_graph_prepared.h` (the lease is checked once per run, not per parallel job:
`prepared_par_lease_ok`), and `prepared_hmx_conv.inc` (divide-free `ktab`; fi and
the finish record are built in near and vector-copied to VTCM; `HMX_SETUP_SLOT`
macro, default 2). Run the full gate (below). If it passes, commit both.

Current time split (ms, Frame, `frame/build/ab/h8/table.txt` is the probe build):
non-conv 156; HMX MAC 45; HMX setup ~23 (checks 5, requant/wsum 6,
tables/fi/tyx 10, parallel ready+enter 1.4); A-tile prep ~20 (34 with the
HVX-path convs); quant ~8; u16 convs (HVX) ~52; stride-2 6/36 and small 1x1
6/32 (HVX path) ~35; executor + VTCM copy ~12.

## How to test (every step ends at all of these)

1. Host, 2 s per run, scalar model of the HMX path:
   `sh .work/merge/build_host.sh && QD_HMX=1 QD_FUSE={0,1} QD_NEAR=1 QD_PAR={0,1,2,3} .work/full/test_full {primary,alternate} mixedfast whole`
   (QD_PAR=3 means real pthreads. Fuse with PAR=3 *without* HMX has a pre-existing host-only race: shared
   scalar fuse buffers. Ignore it.)
2. V75 sim, about 25 min, the real HVX+HMX code with pool 1 and the workspace in VTCM as on the Frame:
   `SEGDIR=.work/full/seg-X sh frame/native_program/run_prepared_reference_sim.sh primary segments 12 fast hmx fuse`
   It must report 12/12 `PASS primary reference segment`. Do **not** edit the engine
   sources while the docker compile runs (until `$SEGDIR/*.log` appear).
3. Frame: `sh frame/bench/ab.sh -t TAG hmx`. It builds, copies, runs the short check plus
   full both models, smoke+random, repeat 2, and prints the table into
   `frame/build/ab/TAG/table.txt`. Use `-q` (primary smoke only) for timing
   iterations. `-n` reuses built ELFs. Never edit sources while ab.sh is building.
4. Census builds: `OUT=.work/full/test_shapes EXTRA=-DCONV_SHAPES sh .work/merge/build_host.sh`
   prints `CONV`, `PARTS` and `HMXCONV` lines. `EXTRA="-DDM_SHAPES -DALL_SHAPES"` prints every op's tensors.

Commit after each passing step, with the Frame numbers in the message.

## HMX infrastructure you will reuse

- `native_program/prepared_hmx_conv.inc`: the current exact path (v6 numerics: u8 in
  the high byte of 16-bit lanes, `qd_hmx_wide_n`, HVX `hmx_quant_pair`). It covers
  tiling (32 output pixels per M tile, P = y*ow+x), the k order matching the in-place
  weight layout (each output block's weights are contiguous 1024-byte K blocks in
  pool 1), row-pointer tables (`hmx_tab`), the fused REQUANT/ADD epilogue, a host
  scalar model, and parts via `nprg2_parallel_for`.
- `Nprg2HmxOps` (`prepared_reference.h`): enter/leave (SSR.XE2), wide,
  wide_n, finish, clock. Frame wiring: `qd_graph_prepared.h` under
  `QDS_PREPARED_HMX`. HMX is already powered and clocked (979.2 MHz,
  AXI 750) for the whole run by `qdrc_power(1)` at init.
- New HMX instructions: edit `frame/qd_hmx_ops.S`, assemble with the SDK in docker
  (`hexagon-clang -mv75 -mhmx -c qd_hmx_ops.S -o build/qd_hmx_ops_native.o`), then
  run `python3 frame/bench/encode_hmx.py build/qd_hmx_ops_native.o --source frame/qd_hmx_ops.S --output frame/qd_hmx_ops_encoded.S`
  on the host (the docker image has no python). The engine and the sim both link the encoded file.
- Small HMX unit tests on the sim: `.work/hmxn/t.c` + `var.S` (compares HMX
  variants tile by tile; see the docker one-liner in its history: compile with
  `-mhvx` plus the encoded .S, run `hexagon-sim -mv75`).
- Lessons: integer `cvt.uh = acc(2):2x1` (transfer-and-clear) is **not** exact;
  `mxclracc` + `acc(0)` per tile is. The cost is ~1.3 µs per output tile
  (34.5K tiles = 45 ms), not the bias reload. Hexagon has no hardware
  divide: a `/` or `%` in a hot loop was 25% of prep. Avoid int64 index math in
  hot loops. The HMX eligibility decision depends on data: filter rows are written
  by earlier commands (ops 32–38), so the dry census undercounts HMX convs. The
  run is safe (workspace-bounded), but prefer data-independent decisions.

## Task 1: exact native-rate HMX (QNN's own conversion), the critical step

Why: native int8 HMX processes 64 pixels per activation tile (2x ours), converts
straight to u8 (`cvt.ub`, no HVX quant), and supports `:deep:cm` crouton
loads/stores. Earlier native attempts (v7/v8) were within 0/+1 of the reference per
conv, **one-sided, never −1**. That suggests a systematic rounding/bias offset,
not a capability limit.

Existing material:
- `frame/qd_hmx_native_kernel.h`, `qd_hmx_native_ops.S` (v7/v8), `build_native_hmx.sh`,
  `bench/hmx_native_conv_sim.c`, `check_hmx_native_conv_sim.py`, `bench/hmx_native_conversion_sim.c`.
  `NATIVE_EXECUTOR_HANDOFF.md` §"Validated hybrid result" has the measured deltas.
- **The original QNN aligned 1x1 HMX leaf**, extracted from the shipped QNN binary:
  `frame/qd_qnn_hmx_1x1.S`, `bench/extract_qnn_hmx.py`, `bench/qnn_hmx_1x1_sim.c`,
  `bench/check_qnn_hmx_1x1_sim.py`. It passes 128 fixtures against an int64 oracle on the sim.
  It shows how QNN programs bias/scale for native conversion. Read its packet
  operands: "bias starts at the supplied pointer, not pointer+256".
- x86 HTP simulator ground truth: `sim/` (`qnn_trace_hook.cpp` LD_PRELOAD per-op
  hook, `kcall.cpp` direct kernel calls), `tools/run_qnn_sim.py`, `tools/trace_qnn_sim.py`,
  `tools/kcall.py`. The project memory note "HTP sim instrumentation" has details.

Steps:
1. Build a per-conv oracle harness. For each real u8 conv shape (census
   `HMXCONV` lines: 3x3 ci 32/64/128/256, 1x1 ci 128), take the int64
   reference `requant_value` (prepared_reference.c) and the v7 native result on the
   V75 sim. Find every differing lane, and tabulate the delta against acc, bias, pre and the
   fi words. Hypothesis to test first: the native conversion rounds where the
   reference floors (or adds a half-LSB). Fix it by adjusting the per-channel
   bias word (e.g. subtract the rounding term, as v6 does with `- (1<<(pre-1))`)
   or by choosing a different cvt mode/scale encoding.
2. Cross-check against the QNN 1x1 leaf: decode exactly which fi/scale/bias words
   and cvt variant it uses for the u8 output, and adopt them.
3. Once a native conversion is exact on every real shape (sim, every lane, plus
   random and boundary fixtures for acc/bias/pre/M ranges: M 1024..2047, pre 2..7,
   post 14), add it to `prepared_hmx_conv.inc` as a second path behind a flag
   (`QDS_PREPARED_HMX=2`). Keep v6 as the fallback. The output must still apply
   the fused epilogue (keep HVX for fused convs at first, or fold the epilogue
   in later).
4. Also attack the per-tile cost on hardware. Write a Frame microbenchmark
   modelled on `bench/hmx_tune.c` / `run_hmx_tune.py` (that harness reached
   12 TFLOPS fp16): time tiles/s for our integer loop vs variants (two
   accumulators, no clear, `:after` store). The sim models no timing, so
   timing needs hardware, and exactness needs the sim.
5. Gate: sim 12/12, then the Frame windows `verify_l2_short --first 360 --last 940`
   (HMX-heavy), then the full A/B.

Expected: the HMX MAC phase drops from 45 ms to ~10–20 ms. Quant moves into HMX.

## Task 2: direct crouton (blocked-layout) HMX loads instead of im2col A tiles

We currently write 9x-expanded A tiles: 701 MB of VTCM writes per inference
(prep ~20 ms + memory traffic). QNN's HMX reads the blocked activations
directly (`:deep:cm` activation loads, `:cm` stores). v8 already uses cm
loads/stores for its tiles. Steps:
1. From the v8 code and the QNN 1x1 leaf, document the cm tile addressing:
   which crouton (8 rows x 8 px x 32 ch u8 = 2048 B) layout one cm activation
   load consumes, and how the range register selects consecutive blocks.
2. For 1x1 convs (ci 128): the input blocks *are* the A operand. Point the cm load at
   the block (one per channel block, per 8-pixel group) without any prep. Output via
   `:cm` store straight into the output blocks (no quant pass if Task 1 gives `cvt.ub`).
3. For 3x3: per tap (ky,kx) the A operand is the input shifted by (ky,kx). A shift
   of 0 in x maps to whole blocks. For kx = 1,2, either keep a small per-row shifted
   copy (1/3 of today's prep traffic, or less), or use a weight-side shift (pad the
   weight with zero taps so all A loads are block-aligned). Watch the known "3x3 HMX stall
   with scattered A pages" (handoff 2026-10-30): keep A pages compact.
4. Blocked layouts in pool 1 are not guaranteed contiguous (block tables), so an A
   K-strip may span non-contiguous blocks: batch per block run, or compact into the
   VTCM tail (the 512 KiB tail is all workspace now).
5. Same gates. Expected: prep goes from ~20 to ~3 ms and quant/store disappears into `:cm`.

## Task 3: non-conv ops and per-command setup (certain wins, ~130 ms)

Non-conv is 156 ms of scalar reference code (`prepared_reference.c`, int64/double
per element over generic `A` arrays; roughly 100–300 cycles/element). Top items
(ms, count): 29/38 DEQUANTIZE 16.6 (x5), 41/72 MUL_INLINE 10.7 (x10), 28/48
QUANTIZE 10.6 (x2), 40/11 ABS 10.2 (x4), 35/23 CONVERT_WEIGHTS 10.2 (x80),
7/31 S2D 9.9 (x174), 23/37 DECIMATE 9.0 (x13), 18/56 + 18/55 TRANSPOSE 13.6,
10/13 ADD_U8 5.7, 24/51 RESIZE 5.3, 8/70 REQUANT_U8 4.5, 39/68 LINEARCLIP 4.0,
25/63 3.4, 16/42 3.0, plus a long tail. Shapes: the `ALL_SHAPES` census.
Opcode names: `qd_resident_schema.h`.

Exactness recipes already worked out:
- DEQUANTIZE (`k_dequantize`): `exact=(v-zp)*sc` is an exact 41-bit product in double;
  the result is RNE but on a **tie it takes the odd mantissa** (rounded vs
  `nextafterf` toward exact). Implement in integer: `P=|v-zp|*m24`, keep 24
  bits, round half to odd, build the float bits. Fall back per element for
  non-normal scales and results.
- QUANTIZE (`k_quantize`): `floor(s*inv+0.5)+zp` with s, inv floats, so the product is
  an exact 48-bit integer times a power of two. Computing it in double is
  exact, and the integer floor `(P + 2^(sh-1)) >> sh` (signed) equals it.
  Inf/NaN go to the existing path.
- MUL_U16 (`mul_u16`) needs 64-bit intermediates (`(e + D*hi) >> 15` with D up
  to 2^31). On HVX, split D into 16-bit halves.
- Prefer folding an op into the producing conv's epilogue where the fusion pass can
  see it (`prepared_fuse.c`: REQUANT_U8/ADD_U8 today; extend to LINEARCLIP/u16).
Validate each op with the per-op HVX handler pattern (`prepared_elementwise.c`,
`run_prepared_ew_new_sim.sh`), then the full gates.

Per-command setup (~23 ms HMX, 7 ms HVX path) is constant across inferences:
the block tables, weights and filter rows are fixed per bound program, but the
filter rows are *written by earlier commands in the first run*, so cache after the
first full run, not at preflight. Cache per command index (fi, multv, ktab,
row tables, mt, eligibility) in a malloc'd DDR region given to
`Nprg2ReferenceConv`. Per run, vector-copy fi into VTCM and the tables into near.
Also skip the redundant run-time `xfer_check`/`blocked_overlap`; preflight
already proved them. Expected saving ~25 ms.

Other conv buckets to queue after 1–3: u16 convs (52 ms HVX), possibly on HMX as
a hi/lo byte split once Task 1 is exact; stride-2 6/36 (26 ms) on HMX (its weight
layout has NO_TILE gaps, so a per-command B copy is needed); overlap HMX of batch
b with prep of b+1 on the followers (double-buffer A/C).

## Hazards (frame-dsp-hazards memory)
Never read QDSP6SS via /dev/mem (resets the SoC). Drain HMX (completion tile)
before returning. Don't dmpause a running DMA. If a callback wedges, check the SMMU
logs and reboot the Frame (sudo password and reboot are authorized: see the
depth-model-realtime memory). Frame: `ssh frame`, working dir
`~/frame-dsp/depth-model/wave1-20261101`, venv `~/frame-dsp/depth-model/.venv`.
