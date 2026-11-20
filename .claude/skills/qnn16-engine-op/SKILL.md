---
name: qnn16-engine-op
description: Make a bit-identical speed change to a QNN16 engine op (hexecutor/native/qnn16 tile/HVX forms such as L2NORM, TRANSPOSE, CONCAT, REDUCE_MEAN, CONV3X3, COPY) and prove it - per-kind fuzz on the ISS, whole-program outputs on real crops, byte-identical rebuild of the deployed engine, then build engine-opt.elf. Use when the user asks to speed up a hand program on the DSP, "profile the keypoint program", or to move an op off the generic path.
---

# Bit-identical QNN16 engine op change

Engine sources: `hexecutor/native/qnn16/` (`exec.c` dispatch and tile hooks, `qnn16_tile_*.c` forms, `qnn16_f32.c`
generic float path, `program.h` op kinds and parameters). The workflow and measured cases are in
`docs/model-development/nir-hand-tracking.md`.

Environment (as for `qnn16-lower`): Docker running, and
```sh
export HEXECUTOR_SDK_ROOT=~/Documents/projects/frame-spatial-runtime/samples/depth-model/.work/hexagon-sdk/6.6.0.0
export HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools
W=samples/hand-tracking/.work/qnn-v75/nirhands-skeleton-full-v2     # current baseline program
```

## 1. Baseline on this tree first

Other agents also change engine ops, so recorded numbers go stale. Before editing, run the baseline program on both
real crops and keep the outputs:
```sh
QNN16_SIM_ELF=base.elf sh hexecutor/build-scripts/test_qnn16_program_sim.sh $W/v2/skel $W/<tag>/base20 $W/real/nhwc20.raw $W/real/valid20.raw
QNN16_SIM_ELF=base.elf sh hexecutor/build-scripts/test_qnn16_program_sim.sh $W/v2/skel $W/<tag>/base50 $W/real/nhwc50.raw $W/real/valid50.raw
```
Find where the time goes with per-op cycles: prefix `QNN16_SIM_CFLAGS=-DQ16_SIM_PER_OP` (one line per op; save the
log). Per-kind totals are always printed (`kind N: ops pcycles`); kind numbers are the enum order in
`hexecutor/native/qnn16/program.h` (also listed in `hexecutor/offline/qnn16_scratch.py`), e.g. 1 COPY, 6 CONV3X3,
19 CONCAT, 22 TRANSPOSE, 29 CONV3X3_S2, 42 L2NORM, 43 REDUCE_MEAN.

Record the baseline source revision and preserve any preexisting working edits that affect the engine build.

## 2. Change the op

- New forms must reproduce the generic path's float32 arithmetic in its order. qf32 HVX lanes are usually faster but
  **not bit-exact** (L2NORM qf32 was 22% faster and moved angles by a median 0.3°); the engine is not built with
  `-mhvx-ieee-fp`.
- Decline (fall back) rather than mis-handle aliasing: e.g. the CONCAT form declines when an input shares a VTCM block
  with the output; partial-tile HMX outputs must restore lanes outside the tensor.
- If staging changes, rerun `qnn16_scratch.py` on the program and `hexecutor/tests/test_qnn16_scratch_poison.py`.

## 3. Gates (all required)

1. **Fuzz every changed kind**, generic path vs tile path on random operands:
   ```sh
   sh hexecutor/build-scripts/test_qnn16_program_sim.sh --fuzz <KIND> --all $W/v2/skel $W/<tag>/fuzz<KIND> $W/real/nhwc20.raw $W/real/valid20.raw
   ```
   `--fuzz` takes an **op kind number** (not a seed); `--all` runs every op of that kind instead of one per operand
   layout. Required result: `PASS qnn16 fuzz` with every `compare op ...` line at `0 differ`. If the program lacks
   the layouts you touched, fuzz another program that has them (e.g. `nirhands-skeleton-best/v2`).
2. **Whole program on both real crops**: all six `out*.f32` byte-identical to the step-1 baselines (`cmp`), and the
   new pcycle total.
3. **`make check`** (host build + ctest + policy) still passes.
4. **Byte-identical rebuild**: build the engine from the saved original sources with the same defines and confirm
   its md5 equals the `engine.elf` deployed on the Frame (`ssh frame md5sum ~/nirhands-skel/run/q16-<name>/engine.elf`).
   This proves the baseline you compared against is what runs on the device.

## 4. Build the timing engine

```sh
QNN16_DEFINES=-DQ16_NO_OP_PROFILE sh hexecutor/build-scripts/build_qnn16.sh
md5 hexecutor/native/build/qnn16/engine.elf
```
Output is always `hexecutor/native/build/qnn16/engine.elf`; copy it out under a new name (`engine-opt.elf`) before
building anything else. Deploy it **beside** the old engine, never over it (see `frame-nirhands-deploy`). Without
`-DQ16_NO_OP_PROFILE` each op costs ~1.5k extra cycles. `build_qnn16.sh` expects `build_tensor.sh` outputs
(bootstrap, chunk probe) to exist for a session; the Frame copies live in `~/hexecutor-hand-viewer/dsp/`.

## 5. Report

pcycles before/after (same tree), per-kind deltas, fuzz kinds and layouts, crop bit-identity, engine md5s, DSP ms
per call once measured on the Frame. Record it in the topic guide or private experiment notes (`frame-handoff-update`).

## Gotchas

- `--compare` is not the gate: its reference is the float no-tile path and it flags many correct CONV3X3 ops.
- `real/sim50` from 2026-11-16 04:48 used different inputs and is not a reference; the full_v2 baselines are
  `$W/l2opt/base{20,50}`, but rerun them on your tree anyway.
- Next known costs in full_v2 (after 2026-11-16): CONV3X3 (168 ops, 11.2M), COPY weight staging (605 ops, 6.9M). For
  the umetrack trunk (24.68M total) COPY is the largest; batching both hands per call should amortise it.
- Open: `swish_qf` fails the SWISH/PW_SWISH fuzz screen by up to ~0.03 absolute (pre-existing, cause unknown).
