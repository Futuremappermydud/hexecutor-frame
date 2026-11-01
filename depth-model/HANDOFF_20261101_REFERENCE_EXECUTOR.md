# Handoff 2026-11-01: full-model reference executor

Nothing in this work is committed. Do not commit unless the user asks.

## Where things stand

Every command in both prepared models now runs through the NPRG v2 executor
(`frame/native_program/prepared_executor.c`) **on the host**. The new reference
kernel (`prepared_reference.c`) is an exact port of every `qd_graph.c` CPU kernel
and covers all 76 (opcode, variant) keys, including the NOP duplicate-pad key
(op 1, variant 2).

Host results (`.work/full/test_full`, about 7 s per run):

| Model | Commands | Reference mode | Mixed mode |
| --- | --- | --- | --- |
| primary | 5068 | lockstep + whole PASS | lockstep + whole PASS |
| alternate | 5040 | lockstep + whole PASS | lockstep + whole PASS |

- **Lockstep:** after every single command, all seven pools are byte-identical
  to `qd_graph`'s CPU backend.
- **Whole:** one preflight over the full range writes no pool, then one run
  executes everything.
- **Both:** both graph outputs match the emulator smoke sha256 values in
  `recovered/<model>/emulator_smoke.json`.
- **Mixed mode:** the 18 existing specialized handlers run, chained over the
  reference kernel. When a specialized validator rejects a command, the
  reference kernel runs it. The remaining 54 keys go straight to the reference.
- **Pool-1 ceiling:** no command writes above the DSP pool-1 active ceiling of
  0x780000.
- **Scratch:** the peak is 1,647,936 bytes (see Task 1).
- **V75 simulator:** the reference kernel reproduces the captured CPU prefixes
  (primary commands 0–131, alternate 0–99) bit for bit; see Task 2 for the
  full-schedule run.

What is **not** proven:

- Nothing has run on hardware.
- Nothing is wired into the Frame runtime.
- Reference convolution is a slow scalar CPU port, so it is a correctness path,
  not a performance path.

## Task 1: cut the reference scratch peak (done)

The census peak was 84,235,648 bytes, set by the 9 Gather (op 20, variant 46)
commands per model. Their dry pass didn't know `axis`, so it sized the result
for the worst axis and then added a broadcast buffer.

All 9 Gather commands per model read `axis` and `indices` from direct pool-5
constants (x is 1×1×4096×80, 256 indices, axis 2). A new helper, `const_value`
in `prepared_reference.c`, reads an element of a direct pool-5 tensor during
the dry pass. Gather (axis), Transpose (perm) and DepthToSpace (factors) use it
to size their results exactly; the old `unknown` worst-case path remains only
for non-constant operands.

Results after the fix:

- Scratch peak is **1,647,936 bytes** in both models (op 6 variant 33, primary
  cmd 3155 and alternate cmd 3468). No command needs more than 16 MiB.
- All four configurations (both models, reference and mixed) still pass
  lockstep and whole-schedule.
- In reference mode, lockstep now caps each command's scratch at exactly the
  bytes its own dry pass measured, so a wet pass that needs more fails. As a
  negative control, a cap one byte smaller fails at primary cmd 5.

## Task 2: V75 simulator run (prefix done; full schedule: see below)

New files: `frame/native_program/test_prepared_reference_sim.c` and
`run_prepared_reference_sim.sh [primary|alternate|both] [prefix|full]`. The
test registers the reference kernel for every key, sizes scratch with
`nprg2_exec_reference_scratch`, preflights the range once (no pool writes),
then runs it once with the pool-1 ceiling at 0x780000. It builds unchanged for
the host and for `-mv75`. The simulator build also uses
`-ffp-contract=off -fno-fast-math -fno-strict-aliasing`.

- **prefix** loads `fixtures/<model>-smoke-prefix-<n>.mpfx` (primary commands
  0–131, alternate 0–99; 25 and 13 reference convolutions respectively).
  All seven pools must match the NativeGraph CPU capture. **V75 simulator:
  PASS for both models**, 4–4.5 minutes per run including the build. Host:
  PASS, 0.1 s.
- **full** starts from zeroed pools plus `constant_pool_5.bin`, runs the whole
  schedule on the native-smoke inputs, and checks both output sha256 values
  against the emulator. Host: PASS for both models, 4.6 s.
  V75 simulator, **with the HVX fast conv path (`fast`)**: **PASS, all 12
  segments** (6 per model). Each segment matches all seven pools and both
  outputs against the host end state, and the host confirms both final outputs
  equal the emulator smoke sha256. Run: `SEGDIR=.work/full/segments-fast sh
  run_prepared_reference_sim.sh both segments 6 fast cycles`, 15:16–15:30
  (14 minutes), log `.work/full/sim_segments_fast.log`.
  V75 simulator, **pure scalar reference conv**: not completed. The segmented
  run (14:13–15:26) was stopped deliberately once the fast run was going; no
  segment had finished or failed. Variants 32–36 are proven on the host in
  reference mode, but not in the simulator beyond the prefixes.

## Task 3: Frame runtime integration

The integration point is `frame/qd_graph_prepared.h`:

- `prepared_open` registers handlers. Add an opt-in config flag that calls
  `nprg2_executor_chain_reference` for each specialized key and
  `nprg2_executor_register_reference_missing` for the rest.
  `test_prepared_full_model.c` (`register_handlers`) has the exact sequence.
  The chain storage must outlive the executor; the test uses 64 slots, and
  18 are needed today.
- `prepared_run` currently passes scratch at 0x30078000, 0x8000 bytes (L2). That
  is far too small for the reference kernel. Reference scratch must be in DDR,
  **not** 0x31000000: that address is VTCM, which is now pool 1. Size it from
  `nprg2_exec_reference_scratch` over the schedule, as the census does.
- `NPRG2_EXEC_REGISTRY_CAPACITY` was raised from 80 to 128. `QdPreparedState`
  embeds the registry, so check its size where it is allocated.

## Task 4: Sign CH stack issue (carried over)

`prepared_sign.c` keeps about 20 KiB of page arrays on the stack and does an
O(n²) overlap check. The plan is to move the arrays into caller scratch and to
sort them for an O(n log n) overlap check. Re-run the Sign host test and
`run_prepared_sign_sim.sh` afterwards.

## Task 5: docs

`STATUS.md`, `frame/native_program/EXECUTOR.md` (new "Full-model reference
kernel" section) and this handoff are updated. Keep the proof limits accurate
as later tasks land: nothing on hardware, nothing in Frame.

## Files

New:

- `frame/native_program/prepared_reference.h` / `.c`: reference kernel,
  validate/run/scratch entry points, `register_reference`,
  `register_reference_missing` and `chain_reference`.
- `frame/native_program/test_prepared_full_model.c`: census, lockstep and
  whole-schedule test.
- `frame/native_program/test_prepared_reference_sim.c` and
  `run_prepared_reference_sim.sh`: prefix and full-schedule test for the host
  and the V75 simulator.

Changed:

- `frame/native_program/prepared_executor.h`: registry capacity raised from 80
  to 128.

Reference-kernel notes:

- Each command runs a dry pass (bounds/limit checks plus scratch measurement,
  no writes), then a wet pass.
- Pool 5 has write limit 0. Pool 1 is limited by `pool1_writable_bytes`.
- Checks that depend on data (indices, shifts, the invscale range) run in the
  wet pass, before that command's writes. Gather axis, Transpose perm and
  DepthToSpace factors are read in the dry pass when they are direct pool-5
  constants (`const_value`); they are in both models.

## Build and run (from `depth-model`)

The CPU reference must be built with
`-ffp-contract=off -fno-fast-math -fno-strict-aliasing`, or the results stop
matching bit for bit.

```sh
N=frame/native_program && mkdir -p .work/full && cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror -I$N \
  $N/test_prepared_full_model.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c \
  $N/prepared_abs.c $N/prepared_addsub_flat.c $N/prepared_block_zap.c $N/prepared_flat_vtcm.c $N/prepared_io.c \
  $N/prepared_mean.c $N/prepared_mul_flat.c $N/prepared_quantize.c $N/prepared_quantize_core.c $N/prepared_reshape_cb.c $N/prepared_sign.c \
  $N/prepared_spill_fill.c frame/qd_graph.c frame/qd_resident.c -lm -pthread -o .work/full/test_full

.work/full/test_full primary   reference both
.work/full/test_full primary   mixed     both
.work/full/test_full alternate reference both
.work/full/test_full alternate mixed     both
```

Prefix and full-schedule test, on the host and in the V75 simulator:

```sh
N=frame/native_program && cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror -I$N \
  $N/test_prepared_reference_sim.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c -lm -o .work/full/test_refsim
.work/full/test_refsim both prefix
.work/full/test_refsim both full

sh frame/native_program/run_prepared_reference_sim.sh both prefix
sh frame/native_program/run_prepared_reference_sim.sh both full
```

Smoke inputs are in `frame/build/native-smoke/{x,y,hidden,t}.raw`. The census
prints the peak command's opcode and variant, and lists the first eight
commands that need more than 16 MiB of scratch (currently none).
