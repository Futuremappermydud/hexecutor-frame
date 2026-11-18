# NIR hands: QNN16 lowering notes (collected 2026-11-17)

These notes gather what the cowfish (`synthetic-hand-render`) sessions learned about lowering SkeletonHandNet
checkpoints to the hexecutor QNN16 engine and running them on the Frame. Until now they were spread across agent
memory and two docs. The detailed, dated record is still the source of truth:

- this repo: `docs/model-development/nirhands-skeleton-integration-handoff-2026-11-13.md` (sections from
  2026-11-14 23:15 on: full_v1, full_v2, combined output, engine-opt);
- cowfish: `stereo_nir_hands/docs/quest_shaped_model_plan_2026-11-16.md` (Step 2 toolchain notes for the
  `trunk="umetrack"` variant).

Some of the tools named below were reported as **not tracked in git** in this worktree (see the Quest-shaped plan's
"Toolchain issues found"). Check `git status` before assuming a script exists on another checkout.

## Pipeline (checkpoint → DSP program)

1. **Checkpoint.** A trained `best.pt` from cowfish `data/runs/<run>/` (pulled from `FutureMapper/<run>` on HF).
   For a speed check before training, `stereo_nir_hands/scripts/count_macs.py --save-untrained` writes an untrained
   checkpoint: simulator cycle counts don't depend on weights.
2. **ONNX export** with this repo's nirhands exporter (`export_nirhands_onnx.py`). Check the decoded output against
   PyTorch (full_v2: 5.5e-7). Output goes in `hand-tracking/.work/nirhands-skeleton-<name>/`.
3. **`compile_qnn.py`** in the docker image `hexecutor-qairt-tools` (QAIRT 2.50) → context binary. 96 px models need
   `--input-dim crops=1,2,96,96 --input-dim second_valid=1`.
4. **`htp_graph.py --no-source-names`**.
5. **`trace.sh`** (graph trace). This is not a repo tool: each `hand-tracking/.work/qnn-v75/<model>/` directory has its
   own copy (with its `hook/qnn_trace_hook.so` and `trace/list.txt`), copied from the previous model's directory.
6. **`qnn16_lower.py --write-tensor-config`**.
7. **`trace.sh tensors`**.
8. **`qnn16_program.py`** `--tensors trace/tensors.bin` → `v2/skel-unstaged.*` (takes `--context-binary` if it can't
   find the `.bin` by graph SHA-256).
9. **`qnn16_scratch.py IN.q16p OUT.q16p [CONSTANTS.q16c]`** (positional; assigns HMX staging blocks:
   `skel-unstaged.q16p` → `skel.q16p`). The program's scratch size for full_v2 is 4,031,360.
10. **Simulator check**: `test_qnn16_program_sim.sh` runs the program on the ISS and writes `out<N>.f32`; it does not
    compare with ONNX Runtime itself, so compare those outputs with ORT separately (no checked-in script does this). When engine kernels change, add
    `--fuzz <KIND> --all`: KIND is an op-kind number from `hexecutor/native/qnn16/program.h` (not a seed), and `--all`
    runs every op of that kind instead of one per operand layout. Per-op cycles: `QNN16_SIM_CFLAGS=-DQ16_SIM_PER_OP`.

Steps 6 and 8 (`qnn16_lower.py`, `qnn16_program.py`) take `--known depth-model/sim/qairt_2_50_functions.json`; the
other steps have no such flag. Output lands in `hand-tracking/.work/qnn-v75/nirhands-skeleton-<name>/<ver>/skel.*`.

Simulator environment: `HEXECUTOR_SDK_ROOT=~/Documents/projects/frame-spatial-runtime/depth-model/.work/hexagon-sdk/6.6.0.0`
and `HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools`.

## Gotchas

- **Trace crops were all zeros.** `trace/crops.raw` in the tight96, ft4-fg and full-v2 directories is zero, so older
  "ISS vs ONNX" numbers for 96 px programs were measured on a degenerate input. Always compare on real crops:
  `hand-tracking/.work/qnn-v75/nirhands-skeleton-full-v2/real/nhwc{20,50}.raw` (captured from a live recording).
- **Q16P inputs are NHWC.**
- **No max-pool op** in the QNN16 engine (`MaxPool_valid` has no handler). New architectures use a stride-2 3×3 conv
  stem instead.
- **Float32 fallback path is very slow.** The tight96 lowering once cost 293M cycles there before it was fixed. Check
  that new op shapes (stride-2 shortcuts, upsample + concat) don't land on it.
- **Indirect constants in pool 2.** QAIRT puts the 12-wide soft-argmax grids in pool 2; `qnn16_lower.py` and
  `qnn16_program.py` originally handled indirect constants only in pool 3 (patched 2026-11-16).
- **Fused tanh table.** The scale head's conv gets a direct pool-2 copy of QAIRT's 14-half tanh table that the hook
  can't capture; `qnn16_program.py` reads it from the context binary when the capture fails validation (fixed
  2026-11-16 evening; full_v2 still relowers byte-identical).
- **BatchNorm models in the exporter's PyTorch check** were left in train mode; fixed with `.eval()` (ONNX was fine).
- **`--compare` false alarms.** It flags many CONV3X3/CONV3X3_S2 ops, but its reference is the float no-tile path
  (`Q16_EXEC_NO_TILE`); final outputs matched ONNX end to end. Judge by end-to-end outputs on real crops.
- **Open (umetrack trunk):** `disparity_hm` and `visibility_logit` are ~10× further from ONNX than QNN's own FP16.
  Judge on a trained checkpoint, not an untrained one (flat heatmaps exaggerate it).
- GroupNorm can't be folded, so int8 needs `--norm batch` models. The user reported int8/PTQ barely sped up the
  current model: convs are only ~22% of its DSP cycles. Whether hexecutor has an int8 path at all is open.

## Engine performance work

- Engine kernel changes must stay **bit-identical**: `test_qnn16_program_sim.sh --fuzz <KIND> --all` for every changed
  op kind with 0 differing elements (the 2026-11-16 change used kinds 42 L2NORM and 22 TRANSPOSE), plus whole-program outputs on both real crops, plus proving the
  original sources rebuild the deployed `engine.elf` byte for byte.
- 2026-11-16: L2NORM scalar path for channel counts not a multiple of 32, and a TRANSPOSE fix: full_v2 72.70M →
  50.76M pcycles, 33.0 → 18.2 ms per call on the DSP (`engine-opt.elf`). qf32 lanes were faster but not bit-exact.
- Next costs in full_v2: CONV3X3 (168 ops, 11.2M), COPY weight staging (605 ops, 6.9M). For the umetrack trunk
  (24.68M), COPY is the largest item; batching both hands per call should amortise it.
- **Other agents also optimise engine ops**: rerun the full_v2 baseline on the same tree before comparing numbers.

## Deploying and measuring on the Frame

- Layout: `~/nirhands-skel/run/models-<name>` and `run/q16-<name>` (detector + `engine.elf` / `engine-opt.elf`),
  `run/in-<name>` with the real NHWC crops and their ISS outputs, check scripts like `check_full_v2.sh`
  (bit-compare DSP vs ISS), launch scripts `run_viewer_full_v2*.sh` (`_combined`, `_combined_opt`, `_combined_cpu`).
- `cdsp_mem` needs `sudo insmod` after a reboot, and live capture needs sudo. The user types the password, so
  agents can't start live sessions. The user's group can open `/dev/cdsp`.
- Keep DSP sessions short: the hexecutor docs warn that long firmware wakes can hang the DSP.
- **Headless latency without sudo:** replay `~/nirhands-skel/rec/*.raw.zst` through `/tmp/nirbench/nir_step_bench`
  (source `nir_step_bench.cpp` beside it; `/tmp` is lost on reboot).
- Viewer build on the Frame: `make bin/frame_nir_hand_viewer NIR_ORT=/home/steamos/nirhands-viewer/ort` (the
  Makefile's default ORT path doesn't exist there).
- The user wants the QNN path, not ORT: the ORT CPU path is faster (26 ms step p50) but loads CPUs that overlap
  vrcompositor.
- Remaining latency: the one-second wake rotation adds ~50 ms to about one call in 13 (step p99 129 ms).
