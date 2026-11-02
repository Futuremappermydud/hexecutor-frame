# Handoff: 2026-11-02 session (341 -> 188 ms)

Frame steady ms, exact on both models (smoke + random, repeat 2), all three
gates (host, V75 sim 12 segments, Frame ab.sh) per commit:

| Commit | Change | emulator (`hmx`) | silicon (`hw`) |
|---|---|---:|---:|
| c493eb8 | handoff's uncommitted HMX setup work | 327.1 | |
| 4f49f5e | L2-TCM strips, integer (de)quantize, dry-pass memo | 253.3 | |
| 70b1b7c | stride-2 on HMX, conv setup memo/table derivation | 226.3 | |
| e18c162 | parallel ew/dm at 8K, parallel strips, COPY_VALUES runs | 203.3 | |
| 06731c1 | NPRG2_HW_ROUNDING silicon reference + oracle + gates | 203.2 | 206.4 |
| ec6da41 | native int8 HMX (64-px crouton tiles, cvt.ub :cm) | | 200.7 |
| 7e4f0c5 | fuse cache, native table/prep tweaks | 208.0* | 198.5 |
| 8911e77 | 6/32 (ci 12/20) on HMX via zero-padded K | 200.5 | **188.2** |

(*device-state noise: all ops uniformly 3-7% slower that run.)

## Two exactness targets now

- `hmx` (default): bit-exact vs the emulator hashes (v6 HMX, 32 px/tile).
- `hw` (`QDS_EXTRA_CFLAGS=-DNPRG2_HW_ROUNDING=1`, ab.sh variant `hw`): bit-exact
  vs `recovered/<model>/hw_{smoke,random}.json`, the V75-silicon-rounding
  oracle from the host scalar reference (`tools/make_hw_oracle.py`). Enables
  the native int8 HMX path. Gates: host `QD_EXPECT=<hex0>,<hex1>` (hashes in
  test_prepared_reference_sim.c kExpect), sim `CFLAGS_EXTRA=-DNPRG2_HW_ROUNDING=1`,
  verifier `--expect hw` (ab.sh passes it automatically).
- Background: HMX_NATIVE_ROUNDING.md (silicon = midpoint (2q+1)*M; the emulator's
  mxconvert_to_u8/u16 = q*M; Model is V69 and runs QNN on silicon, so it is not
  emulator-exact either). HW_SEMANTICS_PLAN.md.

## Where the 188 ms goes (t9 hw, handler ms)

conv 6/33 43.1 (x629, 68.5 us/call), 6/35 u16 HVX 23.1, 6/32 11.6, 6/36 5.9,
6/34 4.1; non-conv ~90 (top: 35/23 6.3, 7/31 6.1, 29/38 5.9, 18/56 5.7,
18/55 4.8, 24/51 4.0, 25/63 3.4, 10/13 3.3); dispatch 8.1.
u8 conv phases (cpu, all parts): MAC 3.9, A prep 18.0, setup 20.1, output 1.8.

Profiling builds (HMX_PROF=k charges step k to the u8 output slot; diff vs
baseline): per ~1066 conv commands, step 9 (reference_run -> k_hmx_conv, i.e.
setup()/decode/memo) 8.9 ms, step 5 (k_hmx_conv -> hmx_conv) 6.0 ms, step 2
(filter rows/requant/wsum) 8.0 ms. setup() runs for every reference command,
so the generic per-command cost is ~20-30 ms model-wide. DEQUANTIZE (29/38,
1.2 ms/call, 36 ns/element with 3 parts) takes the integer path for every
element: its cost is the strip machinery/scalar loop on the DSP, unexplained.

## Next (by size)

1. Per-command interpreter cost (setup()+decode ~8 us/command): bind each
   command once (resolved tensors/addresses) after the first run.
2. u16 convs on HMX (6/35+6/34, ~27 ms): u16 crouton block == the :2x1 A tile
   and the cvt.uh output layout (HMX_NATIVE_ROUNDING.md). Blocker: silicon
   computes (acc>>8)+bias, the reference floor((acc + 256*bias - zp*wsum)/256);
   the low byte of zp*wsum needs an extra K lane.
3. Conv setup steps 2/5 (~14 ms): requant/wsum per command; caching needs to
   know which convs have static weights (35/23 CONVERT_WEIGHTS writes the
   dynamic ones each run).
4. Non-conv HVX kernels for the top ops; find DEQUANTIZE's 36 ns/element.

Tools added: NAT_CHECK (native vs v6 per conv, prints first diff), HMX_PROF
steps 1-11, DM_SHAPES "path FAST/generic" census, QD_REPEAT/QD_INPUTS/QD_EXPECT
host env, .work/nat ISS unit test of qd_hmx_native_n (outputs must be
2048-aligned for :cm stores; valign shifts are mod 128).
