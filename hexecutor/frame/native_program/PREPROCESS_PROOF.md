# Prepared preprocess proof record

The new `prepared_preprocess.c/.h` module has two fail-closed handlers:

- InputSlice variant 1, `*InputSlice@Ff.s4*6.`, exact shape-value tuple from command 0. It copies caller IO index 3's 128-byte float32 value into its direct pool-1 output.
- Im2col variant 24, `ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t`, exact u8 `[1,10,258,1]` input, `[3,3,1,32]` kernel, `[1,1,1,1]` stride, and CB `[1,8,256,9]` / max `[1,8,256,32]` output profile. It uses the audited direct im2col helper, only when the 32-page output slice is valid, unique, and disjoint from the input.

Both run only with caller-bound pools/IO and no fallback, heap allocation, or scratch. The required registration call is `nprg2_executor_register_preprocess()`.

## V75 simulator result

`results/preprocess_v75_smoke.log` records the completed strict Hexagon build and simulator run. It compares every byte in all seven pools against actual NativeGraph CPU output after the one target command. It also confirms fail-before-write behavior for an aliased InputSlice input and an Im2col output block redirected to the reserved pool-1 tail.

The alternate Im2col block table contains repeated addresses outside the active 32-page slice. Its `slice_origin=35` range is unique and matches the CPU step; the handler accepts that slice after validating only the active output pages.

No device or latency claim is made. `verify_preprocess_frame.py` is an offline-ready Frame benchmark draft for smoke/random CPU-prefix comparisons; the parent is copying/adapting it into `frame/bench` for the target engine.

## SHA-256 inventory

Generated with `shasum -a 256` after the passing simulator run:

```text
a8b4bad3eef94b5f4a6495210a3f9058e843387dd642e1aba4438ec3de58bafd  prepared_preprocess.c
96e9b535c3daabffad43174f3b0eb90a5ada9b72f5b2114f5b659a9af5f0413a  prepared_preprocess.h
3c3e67bbdc8ac0ff558e5dbe40fc99c0acc602e961055bd21afbe1e247b32042  test_prepared_preprocess.c
c509ace990247d356e05300adb92b9ef7d6ad7b8d642bdc64981f9437508d64e  capture_preprocess_cpu_prefix.py
ffa526fd1ce6226be73cd2abee321055b9ebc5b8a2de160010a37edf79e4cb74  verify_preprocess_frame.py
b31847cfcb5e8db33d4288bdae54a04353eba81d85ad5809729823e317f59b17  run_prepared_preprocess_sim.sh
978e9f8d20134e8fdfb613443706b8384d6d0912bed932bdc0532e4e4998118e  fixtures/alternate-smoke-op3.ppcpu
ed7f21af6a97f0971c60370e98d7a64663fe6734afc69306b2d503c0ce8123e7  fixtures/alternate-smoke-op5.ppcpu
b01dee6a4a9003cbe0010ce63800c9d7f02410adbbc05b087d8657cdabf6b8eb  fixtures/primary-smoke-op3.ppcpu
15c7088c9b0ccf4ad16f58022285f9428243093b60ded4c5064a1447bc65aabc  fixtures/primary-smoke-op5.ppcpu
dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b  ../build/resident-primary.qdrx
54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298  ../build/resident-alternate.qdrx
c6ae3022dbbdab04f474751bd987e9bfc0547337d5e9b95becb2e100eed86013  examples/primary-v2.nprg
21953104d4960e19dff0f61116dd787e2eb533ef1e4c8eeeedfba197787f69a4  examples/alternate-v2.nprg
```
