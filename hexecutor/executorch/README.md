# HexecutorBackend (ExecuTorch delegate, C++ runtime)

`HexecutorBackend` is an ExecuTorch `BackendInterface` that runs a delegate
blob on the hexecutor `tensor_f32` backend (an HXFG image plus its constant
pool). This directory holds the C++17 runtime half. The Python AOT half
produces the blob; it is not built or run here.

## Files

- `blob.hpp`, `blob.cpp`: bounds-checked parser for blob format v1 and the
  NCHW/NHWC transpose helpers. No ExecuTorch dependency.
- `hexecutor_backend.hpp`, `hexecutor_backend.cpp`: the backend. Registers as
  `HexecutorBackend` from a static initializer.
- `test_blob.cpp`: parser and transpose tests. Needs no ExecuTorch.
- `hexecutor_et_selftest.cpp`: drives the registered backend directly on an
  existing HXFG image (no `.pte`). Checks NHWC/NCHW equivalence and shape rejection.
- `hexecutor_et_run.cpp`: loads a `.pte`, runs `forward` on seeded random
  float32 inputs, prints output summaries.
- `CMakeLists.txt`: standalone project. It references the existing hexecutor
  host sources and `hexecutor/host/cpp/tensor_reference.cpp` by path. It does
  not edit any existing CMake file.

## Blob format v1 (little-endian, no padding)

```
"HXET" u32 version=1 u32 backend=1 (tensor_f32) u32 n_inputs u32 n_outputs
per input, then per output:  u32 layout  u32 rank  u32 dims[rank]
u64 image_size  u64 constants_size  image bytes (HXFG)  constants bytes
```

- `layout` 0: the tensor is passed as-is.
- `layout` 1: PyTorch NCHW. Inputs are transposed NCHW to NHWC before the run,
  outputs NHWC to NCHW after. Rank must be 4.
- `dims` are PyTorch-side dimensions. Every dim must be nonzero and the byte
  size must not overflow. Rank is 1 to 8. At most 256 inputs and 256 outputs.
- The parser rejects bad magic, version or backend, unknown layouts, truncation,
  overflow, oversized section lengths and trailing bytes.

## Build

ExecuTorch's CMake requires its source directory to be named exactly
`executorch`, and it resolves symlinks, so the clone must sit in a real
directory with that name. Submodules needed by the minimal build:
`third-party/flatbuffers`, `json`, `gflags`, `flatcc`, and
`backends/xnnpack/third-party/FXdiv` (CMake adds it unconditionally).

```sh
cd hexecutor/.work
mkdir -p et-src
git clone --depth 1 --branch v1.0.0 https://github.com/pytorch/executorch et-src/executorch
(cd et-src/executorch && git submodule update --init --depth 1 \
    third-party/flatbuffers third-party/json third-party/gflags third-party/flatcc \
    backends/xnnpack/third-party/FXdiv)
ln -sfn et-src/executorch executorch-src   # optional convenience alias

# Blob tests only (no ExecuTorch):
mkdir -p blob-test
c++ -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined \
    -o blob-test/test_blob ../executorch/test_blob.cpp ../executorch/blob.cpp
blob-test/test_blob

# Full build (ExecuTorch runtime + backend + tools). The configure step builds
# ExecuTorch's CMake tree and takes several minutes on a laptop.
cd ../..
cmake -S hexecutor/executorch -B hexecutor/.work/executorch-build -DCMAKE_BUILD_TYPE=Release \
      -DHEXECUTOR_ET_SOURCE_DIR=$PWD/hexecutor/.work/et-src/executorch
cmake --build hexecutor/.work/executorch-build -j8 --target hexecutor_et_selftest hexecutor_et_run
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `HEXECUTOR_ET_REFERENCE=1` | Run the portable CPU reference (`tensor_reference`). Explicit only; there is no automatic fallback. Reads no device variables. |
| `HEXECUTOR_LOADER`, `HEXECUTOR_ENGINE`, `HEXECUTOR_BOOTSTRAP`, `HEXECUTOR_CHUNK_PROBE` | Required in device mode. Explicit artifact paths; there are no defaults. Same roles as `hexecutor_tensor_run`. |
| `HEXECUTOR_TENSOR_F32=1` | FP32 intermediates (`Artifacts::tensor_fp16=false`). |
| `HEXECUTOR_TENSOR_PRECISE=1` | Float DSP path (`Artifacts::tensor_hmx=false`). |
| `HEXECUTOR_MAX_WAKE_SECONDS` | Wake hold limit, as in `hexecutor_tensor_run`. |

Any unset required variable makes `init()` fail with `Error::InvalidArgument`.
Nothing is substituted silently.

## Runtime behavior

- `init()` parses the blob, writes the HXFG image and constants into a private
  `mkdtemp` directory (mode 0700), loads the program with
  `hexecutor::load_tensor_program`, and checks it against the blob contract:
  count, float32 type, and shape (NHWC for layout 1, exact dims for layout 0).
  In device mode it then opens one `hexecutor::Session`.
- `execute()` validates every argument (float32, contract shape) before any
  device work. Arguments are inputs first, then outputs. It packs inputs, runs
  the session, checks output byte sizes, and unpacks into the output tensors.
  Calls on one handle are serialized by a mutex.
- `destroy()` closes the session and removes the temp directory. Reference mode
  reads the temp files on every run, so they stay until `destroy()`.
- Errors return `executorch::runtime::Error` codes. Nothing aborts.

## Verified on macOS (reference path only)

Built with `cmake --build hexecutor/.work/executorch-build --target hexecutor_blob_test hexecutor_et_selftest hexecutor_et_run`
(Release, ExecuTorch v1.0.0). The project's own sources produce no warnings. Warnings from ExecuTorch headers are expected.

- `hexecutor_blob_test`: `blob tests passed`. Covers a valid blob, every truncation, bad magic/version/backend, overflow, trailing bytes, and the transpose round trip.
- `HEXECUTOR_ET_REFERENCE=1 hexecutor_et_selftest hand-tracking/.work/single45.hxfg hand-tracking/.work/landmark.bin single45-input.f32 single45-oracle.f32`: outputs match the stored oracle exactly (max abs diff 0 over 9408 values). NCHW input equals NHWC input, and a wrong shape returns `InvalidArgument`.
- `HEXECUTOR_ET_REFERENCE=1 hexecutor_et_run <a .pte with a HexecutorBackend delegate>`: runs through the ExecuTorch runtime and the backend. Output 1x8x32x32.
- Without `HEXECUTOR_ET_REFERENCE`, and without the `HEXECUTOR_*` artifact variables, the same `.pte` fails cleanly with exit 1 and leaves no temp directory.

Not run here: any device-mode session. The Frame DSP transport is unavailable on this Mac. ExecuTorch's `ET_LOG` is compiled out in this Release build (`EXECUTORCH_ENABLE_LOGGING` is off), so failures show only the `Error` code. Set `-DEXECUTORCH_ENABLE_LOGGING=ON` on the ExecuTorch configure to see reasons.

## Limits

- One `Session` per delegate handle. Merging delegates into one multi-program
  Session or routing through the `hexecutord` broker is future work.
- Float32 I/O only. Batch size and graph shape are fixed by the blob.
- The CPU reference copies the image and constants from disk on every run, so
  it is for correctness checks only.
- The device path needs the Frame DSP transport and cannot run on this Mac.
  Only the reference path is exercised here.
- The backend's static registrar is in an archive. Executables must call
  `hexecutor_et::ensure_registered()` (the runner and self-test do) or link the
  archive whole, or the registrar object may be dropped.
- The ExecuTorch configure takes about 5 minutes on this Mac. Changing
  `CMakeLists.txt` in this project regenerates it, so batch CMake edits.

## Frame results (2026-11-17)

Built natively on the headset (`~/hexecutor-et`, `cmake --build build -j6`) against ExecuTorch v1.0.0. Artifacts: `~/hexecutor-hand-viewer/dsp/` (`qd_loader.elf`, `engine-planar.elf` d6b96ed2…, `bootstrap4.elf` b8ddcf62…, `chunk_probe.elf`). `cdsp_mem` 128 MB (needed `drop_caches` + `compact_memory` before `insmod` after 8 h uptime: the first attempt returned ENOMEM).

| Run | Result |
| --- | --- |
| `hexecutor_blob_test` | pass |
| landmark `single45`, reference | max abs diff 0 vs oracle |
| landmark, DSP float path (`HEXECUTOR_TENSOR_PRECISE=1`) | 4.8e-6 vs oracle |
| landmark, DSP HMX FP16 (default) and `HEXECUTOR_TENSOR_F32=1` | 2.17e-3 vs oracle (HMX FP16 arithmetic, not intermediate storage) |
| `tiny.pte` through ExecuTorch runtime, DSP vs reference | output sum 4414.926 vs 4414.885; first values agree to ~1e-4 |

`execute_ms` from `hexecutor_et_run` (20.6 ms DSP vs 0.69 ms CPU for `tiny.pte`) is one cold execute including staging, not a warm benchmark. DSP control was `(1,0,1)`, callback 0, regions zero and no eye clients before and after every run.

## Speed test (Frame, 2026-11-17)

`hexecutor_et_run MODEL.pte --warmup 5 --iters 100` (warm `method.execute()` wall time through the ExecuTorch runtime; the first execute and `load_method` are reported separately). Models: torchvision MobileNetV2 backbone (`features`), random weights with BatchNorm statistics calibrated on random inputs and folded into the convolutions (`torch.fx.experimental.optimization.fuse`) before export, fully delegated. CPU column is the single-thread portable `tensor_reference` (not XNNPACK, so not a fair CPU baseline). Errors are DSP FP16 vs that reference.

| Model | DSP median (p90) | CPU ref median | Mean abs err / mean abs ref | Cosine |
| --- | --- | --- | --- | --- |
| tiny (conv/dw/add/sigmoid, 32 px) | 0.54 ms (0.77) | 0.81 ms | 1.1e-4 / 0.54 | 1.000000 |
| MobileNetV2 96 px | 0.45 ms (0.48) | 85 ms | 0.0096 / 0.31 | 0.9995 |
| MobileNetV2 128 px | 0.55 ms (0.63) | 92 ms | 0.016 / 0.34 | 0.9990 |
| MobileNetV2 160 px | 0.79 ms (1.99) | 137 ms | 0.015 / 0.37 | 0.9991 |
| MobileNetV2 224 px | 27.1 ms (27.5) | 241 ms | 0.018 / 0.39 | 0.9989 |

- ExecuTorch overhead is about 0.2 ms: the 224 px HXFG run directly with `hexecutor_tensor_run` takes 26.9 ms (DSP timer 26.5 ms).
- 224 px falls off the fast path: `blocked_fp16=false` (activation arena above the 7 MiB VTCM limit), so activations stay in DDR. 35 HMX ops (the pointwise convolutions); the top-cycle ops are the early high-resolution layers.
- The DSP float path (`HEXECUTOR_TENSOR_PRECISE=1`) matches the reference to 1.4e-4 at 224 px but takes 618 ms, so the FP16 error is HMX arithmetic, not the delegate.
- First execute after `load_method` is 20–47 ms (resident staging); `load_method` (session open) is 0.2–0.55 s. Occasional 70 ms outliers are the 1 s wake-limit re-wake.
- Not yet run on the Frame: models that leave ops on the CPU. Linking ExecuTorch's portable kernels (`-DHEXECUTOR_ET_PORTABLE_OPS=ON`) needs Python `torchgen` at build time, which the Frame lacks. Delegation also needs BatchNorm folded beforehand; the partitioner does not fold it.
