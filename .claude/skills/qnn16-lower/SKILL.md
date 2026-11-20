---
name: qnn16-lower
description: Lower a NIR-hands (SkeletonHandNet / keypoint) checkpoint to a hexecutor QNN16 program (ONNX export → QAIRT 2.50 context → htp_graph decode → trace → qnn16_lower/qnn16_program/qnn16_scratch) and check it on the Hexagon simulator against ONNX Runtime on real crops. Use when a new or retrained hand model has to run on the Frame CDSP, when the user asks "lower X", "how many cycles is X on the DSP", or before deploying a new q16-<name> program.
---

# Lower a checkpoint to QNN16 and check it on the ISS

The maintained workflow and device evidence are in `docs/model-development/nir-hand-tracking.md`.
Per-model `trace.sh` helpers and logs live in the ignored
`samples/hand-tracking/.work/qnn-v75/<model>/` directories. Lowering and simulator checks are offline.

## Prerequisites

- Docker Desktop running (`docker images` must list `hexecutor-qairt-tools` and `depth-model-hmx-tools`; both are
  local images, `--pull never`).
- QAIRT 2.50 SDK: `/Users/owenelliott/Downloads/qairt/2.50.0.260828`.
- Hexagon SDK for the ISS:
  ```sh
  export HEXECUTOR_SDK_ROOT=~/Documents/projects/frame-spatial-runtime/samples/depth-model/.work/hexagon-sdk/6.6.0.0
  export HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools
  ```
- Checkpoint: cowfish `data/runs/<run>/best.pt` (pull from HF `FutureMapper/<run>` with the cowfish `hf-train-run`
  conventions if missing). For a speed-only check before training, cowfish
  `stereo_nir_hands/scripts/count_macs.py --save-untrained` writes an untrained checkpoint (ISS cycle counts don't
  depend on weights; accuracy numbers from it mean nothing).

All paths below are relative to the repo root; work dirs are under **`samples/hand-tracking/.work/`** (ignored). Use
`<name>` like `full-v2`, and `W=samples/hand-tracking/.work/qnn-v75/nirhands-skeleton-<name>`.

## Steps

1. **ONNX export** (cowfish venv; it imports the cowfish model code):
   ```sh
   NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
   "$NIR/.venv/bin/python" samples/hand-tracking/tools/export_nirhands_onnx.py --project "$NIR" \
     --checkpoint <path/to/best.pt> --output samples/hand-tracking/.work/nirhands-skeleton-<name> \
     2>&1 | tee samples/hand-tracking/.work/nirhands-skeleton-<name>.export.log
   ```
   Flags: `--project` (required), `--output` (required), `--checkpoint`, `--detector`, `--opset` (17). It writes
   `detector.onnx`, `keypoints.onnx`, `manifest.json` and reports ORT-vs-PyTorch differences (full_v2 decoded:
   5.5e-7). It refuses temporal checkpoints. A large difference means stop.

2. **QAIRT context** (inside `hexecutor-qairt-tools`; repo mounted at `/workspace`):
   ```sh
   docker run --rm --platform linux/amd64 -v /Users/owenelliott/Downloads/qairt/2.50.0.260828:/sdk:ro \
     -v "$PWD":/workspace -w /workspace hexecutor-qairt-tools \
     python hexecutor/offline/compile_qnn.py --sdk /sdk \
       --model samples/hand-tracking/.work/nirhands-skeleton-<name>/keypoints.onnx \
       --output $W --name nirhands_skeleton \
       --input-dim crops=1,2,96,96 --input-dim second_valid=1
   ```
   Defaults are the Frame target (`--arch v75 --soc-model 57 --vtcm-mb 8 --hvx-threads 4 --optimization 3`).
   Use `crops=1,2,128,128` for 128 px models. Output: `$W/nirhands_skeleton-v75.bin` plus `compiled-summary.json`
   (records the exact commands).

3. **Decode the context**: `python3 samples/depth-model/tools/htp_graph.py $W/nirhands_skeleton-v75.bin $W/decoded --no-source-names`
   (positional `context output`; no `--known` here).

4. **Trace pass 1**: copy `trace.sh`, `hook/qnn_trace_hook.so` and `trace/list.txt` from the previous model's dir
   (e.g. `nirhands-skeleton-full-v2/`); fix the context `.bin` name in `trace.sh` if `--name` differs. Put input
   files in `$W/trace/` (`crops.raw`, `valid.raw`; `list.txt` is `crops:=/ctx/trace/crops.raw second_valid:=/ctx/trace/valid.raw`).
   Then `sh $W/trace.sh` → `trace/trace.log`, `writes.bin`, `calls.log`, `Result_0/`. If the hook source changed,
   rebuild it with clang++ in `hexecutor-qairt-tools` (see `docs/model-development/mercury-hand-tracking.md`).

5. **Tensor config**:
   ```sh
   python3 hexecutor/offline/qnn16_lower.py $W/decoded $W/trace \
     --known samples/depth-model/sim/qairt_2_50_functions.json --write-tensor-config $W/trace/tensor_config.txt
   ```

6. **Trace pass 2**: `sh $W/trace.sh tensors` → `trace/tensors.bin` (pool-2 constants).

7. **Program**:
   ```sh
   mkdir -p $W/v2
   python3 hexecutor/offline/qnn16_program.py $W/decoded $W/trace --tensors $W/trace/tensors.bin \
     --known samples/depth-model/sim/qairt_2_50_functions.json --output $W/v2/skel-unstaged
   ```
   Add `--context-binary $W/nirhands_skeleton-v75.bin` if it can't find the `.bin` by graph SHA-256 (needed for the
   fused-tanh table fallback).

8. **Scratch staging** (positional: `IN.q16p OUT.q16p [CONSTANTS.q16c]`):
   ```sh
   python3 hexecutor/offline/qnn16_scratch.py $W/v2/skel-unstaged.q16p $W/v2/skel.q16p $W/v2/skel-unstaged.q16c
   cp $W/v2/skel-unstaged.q16c $W/v2/skel.q16c; cp $W/v2/skel-unstaged.q16.json $W/v2/skel.q16.json
   ```
   (The constants are unchanged by staging; full_v2's two `.q16c` files are identical.)
   It prints staged convolutions and any `without staging blocks` (those take the slow float path). The engine's
   scratch limit is `0x7ff000`; full_v2 needs 4,031,360.

9. **ISS run on real crops** (minutes; ~13 min for big programs):
   ```sh
   sh hexecutor/build-scripts/test_qnn16_program_sim.sh $W/v2/skel $W/real/sim20 $W/real/nhwc20.raw $W/real/valid20.raw
   ```
   Usage: `[--parts4|--parts4-rev] [--compare] [--stop N] [--fuzz KIND [--all]] PREFIX OUT_DIR INPUT...`.
   Ends with `PASS qnn16 program sim: N ops, M matrix ops, P pcycles` and per-kind cycle lines (kind numbers are the
   enum in `hexecutor/native/qnn16/program.h`). Per-op cycles: prefix `QNN16_SIM_CFLAGS=-DQ16_SIM_PER_OP`.
   Two ISS runs at once collide on `hexecutor/native/build/qnn16/test_program.elf` unless each sets
   `QNN16_SIM_ELF=<unique>.elf`.

10. **Compare with ONNX Runtime** on the same real crop (NCHW `crops20.raw` for ORT, NHWC `nhwc20.raw` for the
    program). There is no checked-in script for this; write a short numpy/onnxruntime comparison. Map QNN outputs by
    size, not by ONNX order (QNN reorders; compiles have differed in angles/palm order). Reference numbers (full_v2,
    real crops): keypoints 0.0037, disparity 0.0022, visibility 0.016, scale 0.0002, angles 0.0049, palm 0.0002.
    The host interpreter check `python3.14 hexecutor/tests/test_qnn16_program.py PREFIX --run TRACE_DIR --inputs DIR`
    compares against QNN's own outputs (and fails tolerance on visibility for 96 px models although the program is
    closer to ONNX than QNN is).

## Gates before calling it done

- ISS `PASS`, cycle count recorded, no unexpected huge kind lines (float-fallback ops show up as multi-million
  pcycle `CONV3X3`/`CONV3X3_S2` entries; tight96 once cost 293M from two ops).
- Accuracy vs ONNX on **real** crops, from a **trained** checkpoint.
- Record: checkpoint SHA-256, ONNX dir, `$W/v2/skel.*`, ops, scratch, pcycles, ONNX diffs in the topic guide or private experiment notes (see
  `frame-handoff-update`).

## Gotchas

- `trace/crops.raw` in tight96, ft4-fg and full-v2 dirs is **all zeros**; never judge accuracy on it. Real crops:
  `samples/hand-tracking/.work/qnn-v75/nirhands-skeleton-full-v2/real/{nhwc20,nhwc50,crops20,crops50,valid20}.raw`.
- **Q16P inputs are NHWC** `[1,96,96,2]`; NCHW gives garbage without an error.
- No max-pool in the engine (`MaxPool_valid` has no handler): use a stride-2 3×3 conv stem.
- New op shapes (stride-2 shortcuts, upsample + concat, partial tiles) can land on the float32 fallback; check the
  per-kind cycles.
- `qnn16_lower.py --write-tensor-config` skips nodes on several HVX worker rows (multi-thread `Transpose_impl`);
  constants may be captured at a later node.
- `--compare` flags many CONV3X3/CONV3X3_S2 ops because its reference is the float no-tile path; judge by end-to-end
  outputs on real crops.
- Open (umetrack trunk): `disparity_hm`/`visibility_logit` ~10× further from ONNX than QNN's own FP16.
- QNN output order differs from ONNX; `Models::keypoints` (nirhands pipeline) assigns scale/palm/angles by size.
