# Prepared padded 1x1: bounded Frame validation, 2026-10-30

The opt-in padded engine `ee5eb80bd1a6673d7e8e54f8240e4bd3bf8ef6c278e503ef29507d92b86f2835` passed bounded command tests on Frame after the four-case SDK numerical proof in `native_program/PADDED_1X1_CPU_PREFIX_SIM_REPORT_20261030.md`. It is accepted for these bounded prepared calls, not for full-model inference.

The preserving bootstrap was `628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147`. The frozen engine source is `.work/frame-direct-kernels-20261030/native-prepared-padded-first-source/`; all 87 source hashes matched the current shared inputs before execution. The Frame script used `--prepared-padded-weights`, the model's `*-v2.nprg`, and CPU-seeded command ranges.

| Model | Sequences | Cases | Result | Packed W bytes | Output difference |
| --- | --- | --- | --- | ---: | --- |
| Primary | 115, 3416 | smoke, random each | 4/4 pass | 1,024 / 4,096 | only 0 or +1 |
| Alternate | 108, 3364 | smoke, random each | 4/4 pass | 1,024 / 4,096 | only 0 or +1 |

Each call used one prepared native 1x1 leaf, no 3x3 leaf, no fallback, and no native failure. All five compared pools matched outside the target output pages. Every run had 65,536 target bytes and no difference outside the 0/+1 bound. Both reports passed with `exit_errors=[0,0,0]`, lease phase 2, no lease error, and equal independent host VTCM fingerprints. DSP range time was about 2.06–2.84 ms; this includes the 7.5 MiB boundary upload and is not full-model latency.

The same engine passed 3x3 regression at primary sequence 247 and alternate sequence 180, smoke and random (4/4). These staged 221,184 input bytes per call, used one native 3x3 leaf, stayed within the 0/+1 output bound, preserved outside-target pools, and restored the VTCM lease with zero exit errors.

Report files and SHA-256:

- `.work/frame-direct-kernels-20261030/native-prepared-padded-primary-matrix.json`: `27824bc618b26eb81ee4b7df4d350d3f7a80230cd9d635be09d4200405db3da1`
- `.work/frame-direct-kernels-20261030/native-prepared-padded-alternate-matrix.json`: `48c368dab8005d7dd096c833efe1bff8341cabcddb533fd2e9f9d41c40eabb6d`
- `.work/frame-direct-kernels-20261030/native-prepared-padded-3x3-primary-regression.json`: `3f0a7e1ed7d1601b10128d44e566fea6b6cd3b73f2ba63549e1a06aa6c50728f`
- `.work/frame-direct-kernels-20261030/native-prepared-padded-3x3-alternate-regression.json`: `7ad36d71d2d111973e2b45adb86e025bccd56fbf65add13e52485e99a948ebf1`

After the tests, `cdsp_mem` was unloaded. Final read-only state: CTRL `[1,0,1]`, callback `0`, loader/payload/arena all zero, eye client in `S` state. No device job remains.
