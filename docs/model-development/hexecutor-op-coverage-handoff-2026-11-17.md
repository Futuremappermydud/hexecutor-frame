# Hexecutor op coverage and larger CDSP memory, handoff 2026-11-17

Everything here is uncommitted in the `cowrie` worktree (branch `Futuremappermydud/feat-mediapipe-hand-npu`). Durable
descriptions of the ops and lowerings are in [tensor-backend](../hexecutor/tensor-backend.md); this file is the diary and
the to-do list.

## Why

Goal: more models out of the box on `hexecutord` / the `tensor_f32` backend, and nothing blocking an arbitrary new model
before testing and optimising. Start state: 11 HXFG op kinds, QLinear-only ONNX, a 32 MiB constant pool, depth/tensor
engines pinned to a 128 MiB `cdsp_mem` lease.

## What changed

**DSP / reference (`hexecutor/native/tensor/`)**
- Op kinds 11 -> 20: `HX_ACT` (relu, relu6, hardswish, hardsigmoid, silu, tanh, leaky, gelu, exp, sqrt, abs, neg, erf),
  `HX_BINARY` (add/sub/mul, broadcast tensor or per-channel constant, reversible), `HX_AVGPOOL`, `HX_SOFTMAX` (channels),
  `HX_SLICE`, `HX_ZEROINSERT` (ConvTranspose), `HX_TRANSPOSE`, `HX_REDUCE` (mean/sum/max/min), `HX_NORM` (channel/group).
  `HX_RESIZE` nearest flag (`p[2]`), `HX_RESHAPE` NCHW flatten/unflatten flags (`p[0]`), conv/depthwise dilation in `z`.
  Scalar activation semantics are one function, `hx_activation` (`graph.c`).
- FP16 plan: any dense conv (kernel up to 15, stride up to 8, padding, dilation) via an im2col pack
  (`fp16_pack.h`, plain C) feeding the existing matrix strip; depthwise shapes outside the hand-scheduled 3x3/5x5 s1/s2
  run a scalar-lane path; all new ops run in the scalar-lane section of `hx_fp16_op`. K must stay <= 4032 for the FP16 plan.
- Constant region: per-graph pool limit 32 -> 64 MiB (`HXFG_MAX_CONSTANT_BYTES`), region end 96 MiB
  (`HXFG_CONSTANT_REGION_END`); host canaries in [64, 96) MiB skipped when constants occupy them (`executor.cpp`).
  Host and simulator checked only (see Open 2).
- `qd_owned_memory.h`: `QDO_SLOTS` overridable (default 8 = 128 MiB, shipped behaviour unchanged).

**Front ends (`hexecutor/offline/`)**
- `lower_onnx_float.py` (new): float ONNX, BatchNorm folding, SiLU pattern, ReLU fusion, constant folding of shape chains,
  oversized-conv output-channel splitting, grouped conv (slice + conv + concat), the ~48 op handlers listed in its
  docstring, and `fp16_plan` blockers in the manifest. 32 MiB pool check is now 64 MiB.
- ExecuTorch (`executorch/lowering.py`): activations, binary ops, avg pool, nearest resize, softmax, transpose/permute,
  reshape flags, reductions, norms, transposed conv, dilation.
- `lower_tflite.py`: HARD_SWISH/TANH/LEAKY_RELU/RELU/RELU6/SUB/MUL/broadcast ADD/AVERAGE_POOL_2D. **Never run** (no `tflite` package here).

**Memory probe**
- `benchmarks/qd_chunk_map_probe.c`: table moved to `ARENA+8192` (`out[4]==2048` marks it); legacy layout kept for 8 slots.
- `host/cpp/frame_transport.cpp`: `verify_chunk_map` honours `HEXECUTOR_CHUNK_WINDOW_MB` (default behaviour unchanged,
  still `size_mb == 128`). New `hexecutor/host/cpp/chunk_probe_run.cpp` -> `hexecutor_chunk_probe LOADER PROBE_ELF`.
- `offline/cdsp_mem_check.py`: same window option via `CDSP_WINDOW_MB` / `CDSP_PROBE_ELF` (Python path needs numpy; not run).

**Agent work:** four Haiku agents (nearest resize, unary math, softmax, slice/split/grouped conv) then six more
(dilation, ConvTranspose, transpose/NCHW reshape, reductions, normalisation, real PyTorch exports). I reviewed their
reports and gated the combined tree; I did **not** read every diff line by line.

## Checks (all run 2026-11-17)

- `ctest` 16/16 (includes `test_tensor_graph`, new `test_tensor_pack` im2col-vs-direct-conv incl. dilation).
- ONNX tests (`test_lower_onnx_float` 33 + `test_lower_onnx` 3) and ExecuTorch tests (24) pass against a freshly built
  `build/bin/hexecutor_tensor_run`; the numeric ones compare with onnxruntime / torch to ~1e-4 (real exports ~1e-7).
- Hexagon simulator (`sh hexecutor/build-scripts/test_tensor_sim.sh`, env below): all three binaries PASS, including every
  new op, dilated and im2col convs, a multi-batch pack, scalar depthwise. One failure on the first run was a test bug
  (5x5, 64 in-channels overflowed the test's constants buffer); the case now uses 32 channels. Re-run after the
  constant-limit change: PASS.
- `make check` (incl. artifact policy) and `make check-docs` pass.
- Real exports (`hexecutor/tests/real_exports_check.py`, needs torch + torchvision, not auto-run): MobileNetV2, MobileNetV3
  block, tiny U-Net, shape-chain models lower and match onnxruntime; **ResNet-18 fails** (pool 56 MB > limit at that time).
- **Not run:** any op on the Frame DSP (no timings for the new ops), TFLite lowering, `hexecutord` with a new-op graph,
  the 96 MiB constants region on the device.

Simulator env: `HEXECUTOR_SDK_ROOT=~/Documents/projects/frame-spatial-runtime/depth-model/.work/hexagon-sdk/6.6.0.0`,
`HEXECUTOR_TOOLCHAIN_IMAGE=depth-model-hmx-tools`.

## Frame (kernel 7.2.0-g9a8c8ea46950, `ssh frame`, sudo password `frame1`)

The Frame had moved to a new DHCP address; `~/.ssh/config` was updated by the user. Results:
- `insmod ~/frame-dsp/kmod-7.2/cdsp_mem.ko size_mb=512` works (log: `cdsp_mem: 512 MB at CDSP IOVA 0xa0000000`).
- CPU mmap of all 512 MiB: per-page pattern read back, 0 mismatches (`/tmp/cpu_map.py` on the Frame).
- DSP chunk probe with the module at 512, loader `~/frame-dsp/quest-depth/wave1-20261101/frame/build/qd_loader.elf`:

  | Window | JTLB entries | Result | Wall time |
  | --- | --- | --- | --- |
  | 128 MiB (`/tmp/probe_8.elf`) | 64..71 | clean | n/a |
  | 256 MiB (`/tmp/probe_16.elf`) | 64..79 | clean | 0.126 s |
  | 512 MiB (`/tmp/probe_32.elf`) | 64..95 | clean | 0.176 s |

  Run: `cd /tmp/cowrie-src/build && HEXECUTOR_CHUNK_WINDOW_MB=512 ./hexecutor_chunk_probe $LOADER /tmp/probe_32.elf`.
  The probe is read-only; no DSP writes beyond 128 MiB have been tested.
- DSP state after the last run: control `(1,0,1)`, callback `0x0`, loader/payload/arena regions zero.
- **Left as is:** module loaded at **512 MiB**. The shipped depth worker / hand viewer / `hexecutor_tensor_run` call
  `verify_chunk_map` and refuse to start while `size_mb != 128`. To go back: stop clients, then
  `echo frame1 | sudo -S rmmod cdsp_mem && echo frame1 | sudo -S insmod ~/frame-dsp/kmod-7.2/cdsp_mem.ko size_mb=128`
  (then run the chunk map check before any DSP write).
- Files on the Frame (all scratch, safe to delete): `/tmp/cowrie-src` (host sources + a minimal CMake and build),
  `/tmp/probe_{8,16,32}.elf`, `/tmp/window_probe.py` (needs numpy, unusable), `/tmp/cpu_map.py`.
  The probe ELFs are also in `hexecutor/native/build/chunkprobe/` (ignored) on the Mac.

Cost note: the mapper's cache-clean over the window runs on every map and unmap, so cost grows with the window.
Map only what a session needs.

## Still open / next

1. **Make the Frame usable again.** Either reload at 128 (command above) or relax `verify_chunk_map` in
   `frame_transport.cpp` from `size_mb == 128` to `>= 128` (extra lease is unused by the 8-slot engines). Recommended:
   relax it, since the larger lease is harmless for them.
2. **Run the 96 MiB constants region on the device.** The change is host+simulator only. Test with a graph whose
   constants exceed 32 MiB through `hexecutor_tensor_run`; check `check_guards()` and the DSP state afterwards.
3. **Dynamic window (user asked).** Design: module at 512 (or 768) once; session config carries `window_mb`; bootstrap
   installs ceil(window/16 MiB) JTLB entries from a runtime slot count instead of the compile-time `QDO_SLOTS`; default
   stays 128; host mmaps and verifies `lease >= window`; place constants above 128 MiB when needed. Growing the kernel
   lease at runtime is not recommended (contiguity failures; stale-translation hazard after module changes).
   Decision for the user: 512 vs 768 MiB lease.
4. **Constant pool for ResNet-class models.** Each conv is stored FP32 and as packed FP16 (about 2x). Options: drop the FP32
   conv weights for FP16-eligible graphs (halves the pool, loses the FP32 fallback), or use the bigger region from item 3.
5. **Performance.** The new ops, the im2col pack and the scalar depthwise are scalar-lane code: correct, slow. Nothing new
   has been timed on the Frame. Vectorise activations, binary ops, softmax, norms, im2col; fuse hardswish/SiLU into the
   conv epilogue; then time MobileNetV2/ResNet-18-style graphs with `hexecutor_tensor_run`.
6. **Coverage gaps.** Batched MatMul / attention (biggest), full-size constant Add/Sub/Mul operands, K > 4032 dense convs
   (FP32 path only; could split input channels with Slice + Add), dilation on pools and ConvTranspose, ONNX nearest
   Resize only with `coordinate_transformation_mode=asymmetric` and `nearest_mode=floor` (ONNX default is
   `round_prefer_floor`), int tensors / non-float32 Cast, TFLite lowering untested and behind the ONNX one, ExecuTorch lacks
   Slice/Split/grouped conv/`addmm` (nn.Linear after a view stays on the CPU)/`instance_norm`, and a view right after a
   delegated permute stays on the CPU.
7. **Review and commit.** Nothing is committed. Read the agent-written diffs (especially `lower_onnx_float.py`, ~1,500
   lines, and `fp16.c`) before committing. Suggested split: DSP ops + tests, front ends + tests, memory probe, docs.
8. **Test environments are scratch.** The onnx/onnxruntime venv and the torch 2.15 + torchvision export venv lived in the
   session scratchpad and will disappear. Recreate: `uv venv --python 3.12 V && UV_HTTP_TIMEOUT=300 uv pip install --python
   V/bin/python onnx onnxruntime numpy` (slow network; run it in the background); run tests with
   `PYTHONPATH=. V/bin/python -m unittest hexecutor.tests.test_lower_onnx_float`. `HEXECUTOR_TENSOR_RUN=<runner>` overrides
   the reference runner path in the test files (default `build/bin/hexecutor_tensor_run`; rebuild it first, it goes stale).
   The ExecuTorch tests use `hexecutor/.work/executorch-venv`.

User decisions pending: items 1 and 3 (lease size and how dynamic), and whether to commit.
