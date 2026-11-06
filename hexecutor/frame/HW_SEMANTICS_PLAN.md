# Hardware-semantics reference + native-rate HMX (plan, 2026-11-02)

Why: v6 (emulator-exact) HMX runs at 32 px/tile, needs a 9x im2col A
expansion and a separate HVX quant pass; u8 conv 6/33 alone is ~56 ms. Native
int8 HMX (64 px/tile, `cvt.ub`, `:deep:cm` crouton loads, `:cm` stores) cannot
reproduce the emulator (HMX_NATIVE_ROUNDING.md), so the gate moves to a
reference we define: **V75 silicon conversion semantics**, measured exactly on
the ISS. Model (V69) is not the oracle; host model = sim = Frame is.

## Semantics (all convs, every path)

u8 output (post = 14):

    q   = floor((acc + off) >> pre)
    out = clip(((2q + 1) * M + 2^14) >> 15, 0, 255)       # was (q*M + 2^13) >> 14

u16 output (pre = 8, post = 26 - e):

    x   = (acc + 256*bias) >> 8                              # unchanged
    q   = post <= 18 ? x << (18 - post) : x >> (post - 18)   # floor(x * 2^(e-8))
    out = clip(((2q + 1) * M + 2^18) >> 19, 0, 65535)

## Switch

Compile-time `NPRG2_HW_ROUNDING` (0 = emulator, default; 1 = silicon), threaded
through: `requant_value`, `rq_prepare`/`rq_emit` (HVX conv path, u8 and u16),
`hmx_quant_pair` + its scalar host model (v6 HMX), and the new native path.
Fused epilogues (REQUANT_U8/ADD_U8 after the conv) are unchanged: they are HVX
ops in QNN too.

## Oracle and gates in HW mode

1. Host `reference` mode (pure scalar) with the flag writes
   `recovered/<model>/hw_{smoke,random}.json` (sha256 per output).
2. Host fast modes, the V75 sim segments and the Frame verifier compare against
   those (verify_native_graph.cases picks hw_* when the engine reports HW mode).
3. Emulator mode stays buildable and gated as today (regression for v6 code).

## Native path (QDS_PREPARED_HMX=2, HW mode only)

1. u8 1x1 (ci multiple of 32): A = the input crouton blocks themselves
   (`activation.ub = mxmem(..):deep:cm`), no prep; output `mxmem(..):cm = cvt`
   straight into the output blocks when not fused; fused convs keep the HVX
   epilogue on the u8 tile.
2. u8 3x3 / 2x2 (s2): per tap, x-shift 0 is block-aligned; kx = 1, 2 via a
   per-row shifted copy (1/3 of today's A traffic) first, weight-side shift later.
3. ci = 12/20 (6/32): zero-padded B (K = 32), A unchanged.
4. u16: `cvt.uh:2x1` with halfword activations (layout in HMX_NATIVE_ROUNDING.md).

Expected: u8 conv 6/33 56 -> ~20 ms, 6/32+6/36 45 -> ~15, u16 29 -> ~8.
