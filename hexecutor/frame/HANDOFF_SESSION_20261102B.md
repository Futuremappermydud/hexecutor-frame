# Handoff: 2026-11-02 session B (188.2 -> 87.6 ms)

Continues HANDOFF_SESSION_20261102.md. Steady-state ms on the Frame, exact on both
models (smoke + random, repeat 2) for every commit:

| Commit | Change | `l2hw` (silicon rounding) | `l2hmx` (emulator-exact) |
|---|---|---:|---:|
| 8911e77 | start (no L2, 3 threads: `hw` / `hmx`) | 188.2 | 200.5 |
| 492d0be | L2 cache lease + 3 threads (followers on page-7 stacks) | 156.6 | 198.2 |
| 1d224d0 | L2CFG=1: lower 512 KiB stays TCM (stacks, slots, hot program, QdsWork, near) | 128.7 | 140.2 |
| fc87b8f | HVX exact DEQUANTIZE; cancel poll every 32 commands | 121.2 | 134.0 |
| f648c12 | HVX exact QUANTIZE, ABS, CONVERT_WEIGHTS; parallel W<->C transpose | 111.4 | 124.0 |
| 92c7b57 | native u16 HMX convs (cvt.uh:2x1); vector conv setup; inline small memcpy | **87.6** | **117.9** |

First (cold) run is still ~182 ms (hw).

## Gates (run all three per batch)

- Frame: `sh frame/bench/ab.sh -t TAG l2hw l2hmx` (full: both models, smoke+random,
  repeat 2; `-q` = primary smoke only). `l2hw`/`l2hmx` are now the main variants
  (QDS_L2_CACHE=1, threads 3, fuse, HMX; `l2hw` adds -DNPRG2_HW_ROUNDING=1).
  Extra defines in a variant: `QDS_EXTRA_CFLAGS=-DA=1+-DB=2` (`+` separates).
  In zsh, pass variants as separate words (`$V` doesn't split; use `sh -c`).
- Host (3 s per run): build `test_prepared_reference_sim.c` + prepared_reference.c,
  prepared_executor.c, prepared_fuse.c, nprg2_runtime.c with `cc -DNPRG2_HW_ROUNDING=0|1`,
  then run `<bin> primary|alternate split 1 <dir> fast` -> `PASS ... final outputs match`.
- Sim unit/differential tests for new HVX kernels (in `.work/setupbench/`, not committed):
  `dqtest.c` (dequantize, 13.6M cases), `qatest.c` (quantize 8.8M, abs 12.6M),
  `uh9.c` (HMX cvt.uh with KB=9 full-range data), `difftest.c` (real command on
  deterministic random pools: host scalar vs sim HVX pool hashes; exclude NaN inputs,
  because the scalar `(uint16_t)NaN` cast differs by target), `varbench.c` (sim pcycles
  per variant), `shapes.c` (host: dump a variant's tensor shapes), `bench_setup.c`.
  Build pattern: `#include "../../frame/native_program/prepared_reference.c"`, compile in
  the `depth-model-hmx-tools` docker image with the SDK hexagon-clang (see the files).
- The 12-segment sim gate was **not** run this session (Frame + host gates only).

## Key findings

1. **Uncached or cold accesses in per-command and per-job paths were the biggest cost.**
   QdsWork on uncached DDR cost ~8 us per parallel job (field-5 L2 mode); moving it to TCM
   was most of 157 -> 129 ms. The per-command cancel check (two uncached arena loads)
   cost ~3 ms. Check any new per-command code for arena/uncached reads.
2. **L2 layout (qd_l2_cache.h).** QDS_L2_CACHE=1 caches the upper 512 KiB. Lower-half
   map during the lease:
   - 0x30002000: hot program
   - 0x3003c000..0x3006c000: lease stacks for workers 0/1/2
   - 0x3006c000: helper conv slots
   - 0x30074000: QdsWork
   - 0x30078000: conv near
   The expanded block tables are on page 7. The lower half is checked across both mode
   switches; the stacks and QdsWork are excluded from that check. Followers move stacks
   with work op 5 and come back with op 6 (qd_graph_dsp.c `follower_leased`). Field 5
   still builds (QDS_L2_CACHE=5) but is slower.
3. **Per-command setup() is cheap now** (~1.5 us; HMX_PROF=12 measures cold vs warm).
   "Bind each command once" would save only ~3 ms.
4. **The engine builds with -fno-builtin.** prepared_reference.c now maps memcpy/memset
   to `__builtin_*` on Hexagon. Other engine sources still call the range-checking
   memcpy for small copies.
5. **u16 convs:** both variants have input zp 0. Silicon `cvt.uh` is therefore exact
   against the hw oracle, and no correction K lane is needed. If a nonzero-zp u16 conv
   appears: add one K lane with activation 1 and per-channel weight
   w_n ≡ -zp*wsum_n (mod 256), and fold the rest into the bias. That stays exact.
   6/35 is a 3x3 conv with ci 32 and co 2 (padded to 32).
6. **QNN research:** frame/QNN_RESEARCH.md (Sonnet subagent; its ms estimates predate
   this session's gains). QNN doesn't fuse more than our schedule already does. A
   "QNN-like" mode means silicon rounding plus faster kernels, not fewer ops.

## Where the 87.6 ms goes (l2hw, build/ab/b3/table.txt)

- **conv 32.6 ms**
  - 6/33 (629 calls): 22.0 = u8 staging 14.2 + setup 6.3 + MAC 1.9 + output 1.2
  - 6/35: 4.8
  - 6/36: 3.0
  - 6/32: 2.8
- **non-conv 48.3 ms**, a long tail:
  - S2D 7/31: 4.7 (174 calls)
  - TRANSPOSE 18/56: 3.4
  - ADD_U8 10/13: 2.9
  - TRANSPOSE 18/55: 2.7
  - LINEARCLIP 39/68: 2.2
  - REQUANT_U8 8/70: 2.1
  - DEQUANTIZE 29/38: 1.9
  - D2S 25/63: 1.8
  - WEIGHT_SCALES 32/21: 1.7
  - RESIZE 24/51: 1.5
  - MUL_U16 14/74: 1.5
  - MUL_INLINE 41/72: 1.4
  - WEIGHTS_TO_VTCM 4/30: 1.4
  - REQUANT_U16 9/71: 1.3
  - GATHER 20/46: 1.3
- **fixed ~7 ms:** VTCM in/out 2.1, preflight 2-3, dispatch 1.0
- `l2hmx` differs only in conv: v6 HMX 6/33 is 35 ms, and the u16 convs still run on HVX.

## Next (user goal: <50 ms; exact first, then a non-exact QNN-like mode behind a flag)

1. **6/33 staging** (nat_tile/nat_prep_part, 14 ms) and the remaining setup (HMX_PROF
   steps 5/7/9, ~6 ms): per-call A-tile building over 9 taps. Ideas:
   - Reuse A tiles across kx (stage once per ky and shift by valign).
   - Cache hmx_tab per command.
2. **Dynamic-weight chain** (80 per model: transpose / weight scales / convert /
   weights-to-VTCM before each 6/35): fuse into one handler.
3. **S2D (7/31)** into conv A staging, and REQUANT/ADD into conv epilogues where the
   fuse plan doesn't already do it.
4. **Non-conv tail:** use `varbench.c` to find kernels above ~5 cycles/element and
   vectorize them exactly (pattern: HVX path + scalar fallback for rejected lanes, an
   exhaustive sim test, then difftest).
5. **`l2hmx`:** u16 convs on HMX via the v6 (wide_n) path if the emulator-exact build
   matters.
6. **Then the non-bit-exact flag** (exactness-strategy memory): QNN-style kernels with
   their own accuracy check against the oracle (max/mean abs diff on both outputs).

Untested: a failure inside the L2 lease (cleanup path), and the cold-run cost.
