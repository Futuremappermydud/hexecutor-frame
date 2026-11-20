---
name: frame-nirhands-deploy
description: Stage a lowered NIR-hands QNN16 program, its ONNX models and an engine on the Frame (~/nirhands-skel), bit-compare the DSP against the ISS, measure headless latency without sudo, and (re)build frame_nir_hand_viewer / nirhands_replay on the headset. Use after qnn16-lower or qnn16-engine-op, or when the user asks to "put X on the Frame", "how fast is it on the DSP", "rebuild the viewer", or for a launch script they can run.
---

# Deploy and measure NIR hands on the Frame

See `docs/model-development/nir-hand-tracking.md` for the model contract and evidence, and
`docs/runtime/hand-tracking-viewer.md` for the launch interface.

## Rules

- **No live sessions.** Live capture (`stream_nir_pairs`) and `insmod cdsp_mem.ko` need sudo; the user types the
  password. Write the launch script, tell the user the command, and wait for their go-ahead before anything they
  must watch.
- **Short DSP sessions only.** Long firmware wakes have ended at control `(2,0,8)` and needed a reboot
  (`docs/hexecutor/operations.md`). Keep `HEXECUTOR_RUNS` small and `--max-pairs` bounded.
- Check DSP state before and after every DSP run (`frame-dsp-state`). If it isn't `(1,0,1)` with regions zero, stop.
- Never overwrite a deployed engine/program/binary: add a new name (`engine-opt.elf`, `*.pre-<change>`).
- The user's group can open `/dev/cdsp`, so `hexecutor_qnn16_run` and replay work over plain `ssh frame`.

## Layout on the Frame (verify with `ssh frame ls ...`; it can change)

```
~/nirhands-skel/                  rsync of this tree (no .git/.work) + build/
  build/bin/hexecutor_qnn16_run, build/libhexecutor_host.a
  depth-runtime/bin/frame_nir_hand_viewer (old ones kept as *.pre-<change>)
  run/models-<name>/              ONNX export dir (detector.onnx, keypoints.onnx, manifest.json)
  run/q16-<name>/                 nirhands_detector.q16{p,c}, nirhands_keypoints.q16{p,c}, engine.elf, engine-opt.elf
  run/in-<name>/                  real NHWC crops + their ISS outputs
  rec/*.raw.zst                   recordings for headless replay
  check_<name>.sh                 DSP-vs-ISS bit-compare;  run_viewer_<name>*.sh  live launch (user)
~/hexecutor-hand-viewer/dsp/      qd_loader.elf, bootstrap4.elf, chunk_probe.elf
~/nirhands-viewer/ort             ONNX Runtime for aarch64
```
The viewer's `--qnn16-dir` needs the files named `nirhands_detector.*` and `nirhands_keypoints.*`. Detector and
`engine.elf` are usually copied from the previous `q16-*` dir (check md5 against the local build).

## 1. Stage

```sh
N=<name>; W=samples/hand-tracking/.work/qnn-v75/nirhands-skeleton-$N
ssh frame "mkdir -p ~/nirhands-skel/run/q16-$N ~/nirhands-skel/run/models-$N ~/nirhands-skel/run/in-$N"
rsync -av samples/hand-tracking/.work/nirhands-skeleton-$N/ frame:nirhands-skel/run/models-$N/
scp $W/v2/skel.q16p frame:nirhands-skel/run/q16-$N/nirhands_keypoints.q16p
scp $W/v2/skel.q16c frame:nirhands-skel/run/q16-$N/nirhands_keypoints.q16c
scp $W/real/nhwc20.raw $W/real/valid20.raw $W/real/nhwc50.raw $W/real/valid50.raw frame:nirhands-skel/run/in-$N/
# + the ISS outputs to compare against (e.g. $W/real/sim20/out*.f32 -> in-$N/sim20/)
```
If native sources changed (`samples/hand-tracking/nirhands`, `depth-runtime/tools`, `hexecutor/host`), rsync the tree to
`~/nirhands-skel/` excluding `.git` and `.work`, then rebuild (step 4). `depth-runtime/scripts/deploy_nir_models.sh
[LOCAL_DIR] [REMOTE_DIR]` copies a models dir but defaults to `~/nirhands-viewer/models`, not the skel layout.

## 2. DSP bit-compare (program alone)

```sh
ssh frame 'cd ~/nirhands-skel/run && D=~/hexecutor-hand-viewer/dsp && \
  HEXECUTOR_RUNS=5 ../build/bin/hexecutor_qnn16_run $D/qd_loader.elf q16-<name>/engine-opt.elf $D/bootstrap4.elf $D/chunk_probe.elf out-<name>/kp20 \
    --program q16-<name>/nirhands_keypoints in-<name>/nhwc20.raw in-<name>/valid20.raw'
```
Usage: `hexecutor_qnn16_run LOADER ENGINE BOOTSTRAP CHUNK_PROBE OUT_PREFIX --program PREFIX INPUT... [--program ...]`;
env `HEXECUTOR_RUNS` (default 20), `HEXECUTOR_MAX_WAKE_SECONDS`. It prints `dsp ms p50/p90` per program. Gate: every
output `cmp`-identical to the ISS outputs for both crops. Earlier per-model scripts (`check_full_v2.sh`, etc.) wrap
exactly this; copy one for a new name.

## 3. Headless latency (no sudo)

Replay a recording through the full `Pipeline::step` (two hands, detector, decode):
- `nir_step_bench` lives in `/tmp/nirbench/` on the Frame (source `nir_step_bench.cpp` beside it). **It is not in the
  repo and `/tmp` is lost on reboot**; if it's gone, ask the user or rewrite it from `nirhands_replay.cpp`.
- Or use `nirhands_replay` (source in `samples/hand-tracking/nirhands/tools/`): flags `--models --input --max-pairs --threads --quiet
  --qnn16-dir --check-cpu --loader --engine --bootstrap --chunk-probe`. `--check-cpu` compares NPU vs ORT decoded
  outputs per call. Example:
  ```sh
  ./nirhands_replay --models models-<name> --input <pairs.raw> --max-pairs 12 --quiet --qnn16-dir q16-<name> \
    --loader $D/qd_loader.elf --engine q16-<name>/engine-opt.elf --bootstrap $D/bootstrap4.elf --chunk-probe $D/chunk_probe.elf --check-cpu
  ```
Report step p50/p90/p99 and DSP ms per call, against the previous engine measured in the same session. Reference
(full_v2, 2026-11-16): 33.0 → 18.2 ms per call; step p50 72.3 → 41.5 ms; p99 129 ms from the 1 s wake rotation.

## 4. Build on the Frame

```sh
# host library + runner
cd ~/nirhands-skel && cmake -S . -B build && cmake --build build --target hexecutor_qnn16_run
# viewer (Makefile default NIR_ORT path does not exist on the Frame)
cd ~/nirhands-skel/depth-runtime && cp bin/frame_nir_hand_viewer bin/frame_nir_hand_viewer.pre-<change>
make bin/frame_nir_hand_viewer NIR_ORT=/home/steamos/nirhands-viewer/ort
# NPU replay
cd ~/nirhands-skel/samples/hand-tracking && ORT=/home/steamos/nirhands-viewer/ort && \
g++ -O2 -std=c++17 -DNIRHANDS_QNN16 -Inirhands/include -I../../hexecutor/include -I$ORT/include nirhands/src/*.cpp \
  nirhands/tools/nirhands_replay.cpp ../../build/libhexecutor_host.a -L$ORT/lib -lonnxruntime -Wl,-rpath,$ORT/lib -lpthread -o ../../run/nirhands_replay
```
Viewer flags: `--models DIR [--qnn16-dir DIR --loader --engine --bootstrap --chunk-probe] [--threads N] [--seconds N]
[--output skeleton|heatmap|combined] [--heatmap-weight W]`. It needs an FRNIR001 stream (`stream_nir_pairs --raw`).
If `make` complains about OpenXR headers, add `OPENXR_INCLUDE=/home/steamos/frame-spatial-containers/third_party/openxr/include`.

## 5. Live launch (user runs it)

If present, copy the validated baseline launcher `~/nirhands-skel/run_viewer_full_v2_combined_opt.sh` to `run_viewer_<name>.sh`, change the dirs,
and give the user: `ssh -t frame ~/nirhands-skel/run_viewer_<name>.sh [seconds] [record.raw.zst]`. If the Frame
rebooted, they first need `sudo insmod` of `cdsp_mem` (128 MB; `depth-runtime/scripts/setup_frame.sh --qnn` does this and its
chunk check, and uses sudo, so ask first).

The user wants the QNN path, not ORT: the CPU/ORT script (`*_cpu.sh`) is faster (step p50 26 ms) but loads CPUs that
overlap vrcompositor.
