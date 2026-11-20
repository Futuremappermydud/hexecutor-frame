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
  not edit any existing CMake file. With `-DHEXECUTOR_ET_SHARED=ON` it also builds
  `libhexecutor_et` (see "Shared library" below).
- `hexecutor_et_shared.cpp`: the one source of `libhexecutor_et`.

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
| `HEXECUTORD_SOCKET` | Broker socket. If set, the delegate registers its embedded graph there. Without it, an installed broker at `/run/hexecutor/hexecutord.sock` is detected automatically. |
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
- With the installed broker socket (or an explicit `HEXECUTORD_SOCKET`), `init()`
  registers the embedded graph with `hexecutord`; `execute()` submits through
  the broker. The daemon currently has eight registration slots per lifetime.
- Device mode: every delegate handle in the process shares one multi-program
  `hexecutor::Session` (the CDSP has one owner). Handles whose HXFG images are
  byte-identical share one program slot, unless the image has a persistent
  tensor: a stateful program can be loaded once per process. A new distinct
  program closes the session and reopens it with every slot (at most
  `HXFG_MAX_GRAPHS`, 8); if that fails the previous session is restored and
  `init()` fails. Slots stay until the last handle is destroyed, then the
  session closes and the files are removed. Every handle must see the same
  `HEXECUTOR_*` artifacts. Runs from all handles are serialized by one mutex.
- `execute()` validates every argument (float32, contract shape) before any
  device work. Arguments are inputs first, then outputs. It packs inputs, runs
  the session, checks output byte sizes, and unpacks into the output tensors.
  Calls on one handle are serialized by a mutex.
- `destroy()` releases the handle's slot (the last one closes the session) and
  its reference to the temp directory. Reference mode keeps one CPU reference per
  handle and reads the temp files on every run, so they stay until `destroy()`.
- Errors return `executorch::runtime::Error` codes. Nothing aborts.

## Verified on macOS (reference path only)

Built with `cmake --build hexecutor/.work/executorch-build --target hexecutor_blob_test hexecutor_et_selftest hexecutor_et_run`
(Release, ExecuTorch v1.0.0). The project's own sources produce no warnings. Warnings from ExecuTorch headers are expected.

- `hexecutor_blob_test`: `blob tests passed`. Covers a valid blob, every truncation, bad magic/version/backend, overflow, trailing bytes, and the transpose round trip.
- `HEXECUTOR_ET_REFERENCE=1 hexecutor_et_selftest hand-tracking/.work/single45.hxfg hand-tracking/.work/landmark.bin single45-input.f32 single45-oracle.f32`: outputs match the stored oracle exactly (max abs diff 0 over 9408 values). NCHW input equals NHWC input, and a wrong shape returns `InvalidArgument`.
- `HEXECUTOR_ET_REFERENCE=1 hexecutor_et_run <a .pte with a HexecutorBackend delegate>`: runs through the ExecuTorch runtime and the backend. Output 1x8x32x32.
- Without `HEXECUTOR_ET_REFERENCE`, and without the `HEXECUTOR_*` artifact variables, the same `.pte` fails cleanly with exit 1 and leaves no temp directory.

Not run here: any device-mode session. The Frame DSP transport is unavailable on this Mac. ExecuTorch's `ET_LOG` is compiled out in this Release build (`EXECUTORCH_ENABLE_LOGGING` is off), so failures show only the `Error` code. Set `-DEXECUTORCH_ENABLE_LOGGING=ON` on the ExecuTorch configure to see reasons.

## Shared library

`-DHEXECUTOR_ET_SHARED=ON` builds `libhexecutor_et` (`.so.1` on Linux): the backend, the hexecutor host
sources and the ExecuTorch runtime (`executorch_core`, the prim ops, `extension_data_loader`,
`extension_tensor`) folded in as whole archives. A program links only this library and gets the ExecuTorch and
backend headers through its interface. The runtime has to live inside it: the backend registry is a static
object of `executorch_core`, so the process must hold exactly one copy, the one the backend registers into.
Linking the static ExecuTorch archives as well would create a second registry. This is how an MIT or other
non-LGPL program uses hexecutor (see [hexecutor/LICENSE.md](../LICENSE.md)). Steam-Frame-Hand-Tracking's
`mercury/build.sh --executorch` is the first user. On the Mac (2026-11-20), its `fhm_nn_check` linked only
`libhexecutor_et.1.dylib` and ran Mercury's two models as 12 handles in reference mode, within 1.03e-6 of
ONNX Runtime.

## Limits

- One `Session` per process (shared by all handles; see Runtime behavior). Routing through the `hexecutord`
  broker is future work.
- The shared session (2026-11-20) has only run in reference mode on the Mac, where it is not used. Its device
  path (slot sharing, reopen on a new program) has not run on the Frame yet.
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

## Diffusion UNet (Frame, 2026-11-19)

`hexecutor/examples/diffusion/unet2d_export.py`: a diffusers `UNet2DModel` DDPM-style denoiser (32x32x3, channels
32/64/64, one attention block at 16x16 down and up plus the mid block, random weights, 6.4 MiB constants), one
denoising step per `forward(x, t)`. Fully delegated: one delegate, no CPU operators, 265 HXFG ops, `blocked_fp16=true`.
Built in `~/hexecutor-et-diff` on the Frame (current tree, engine `0d0acc5c…` built with `build_tensor.sh` settings,
`cdsp_mem` 128 MB).

| Run | Result |
| --- | --- |
| CPU reference vs PyTorch (Mac, `hexecutor_tensor_run`) | max abs diff 2.2e-6 at t = 999, 500, 10 |
| DSP FP16 vs reference (`hexecutor_et_run --seed 7`) | max 2.4e-3, mean 3.9e-4 (mean abs output 0.29), cosine 0.999999 |
| Warm step through ExecuTorch, DSP | **23.7 ms** median (p90 23.9) over 20 runs; see the table below |
| Warm step, CPU reference on the Frame | 157 ms (single-thread portable reference) |

Speed steps, each checked on the Frame against the reference (cosine 0.999999 throughout, max error about 2e-3):

| Change (engine `fp16.c` unless noted) | Step | What it removed |
| --- | --- | --- |
| baseline (HX_SDPA, scalar lanes) | 638 ms | |
| single-head SDPA as MATMUL / SOFTMAX / MATMUL on token columns (lowering) | 197 ms | SDPA 781 M cycles -> matmul 8 M |
| `norm_group_vector`: HVX group norm, two-pass qf32 statistics | 124 ms | 113 M -> 2 M cycles |
| `act_vector` (SiLU, logistic: exp2 polynomial + Newton reciprocal), `binary_vector` (add/sub/mul/min/max/div-by-constant; same-shape, per-channel tensor or constant) | 78.6 ms | 86 M -> 1.4 M |
| `softmax_vector`: HVX channel softmax | 64 ms | 27 M -> 0.7 M |
| `im2col_tile_3x3`: 3x3 / stride 1 / pad 1 im2col as whole-vector shifts (even width, C % 32 = 0) | 23.7 ms | dense 3x3 conv 78 M -> 4 M (pack was 2-4 MAC/cycle) |

Remaining 43 M cycles are copies in scalar lanes: RESHAPE 12 M (74 ops, mostly the token-view repacks), CONCAT 8 M,
nearest RESIZE 6 M, plus matmul 8 M. Regression check with the previous engine: MobileNetV2 160 px and the hand
landmark graph bit-identical; `tiny.pte` within one FP16 ulp (its sigmoid now runs in HVX), same error against the
reference (max 3.7e-4) and 0.83 -> 0.31 ms. The FP16 simulator test (`test_tensor_fp16_sim.c`) passes.

Rank-2 or rank-3 delegate I/O (such as the `[1, 1]` timestep) used to fail `load_method`: the program reports every
tensor as `[1, h, w, c]` and the delegate compared that exactly with the PyTorch dims. It now compares shapes without
their size-one dims, as `Session` does.

## Mercury hand tracking simulator optimization (2026-11-20)

The two Mercury HXFG exports in `hexecutor/.work/mercury-et/` use a scalar runtime tensor
(`useLastKeypoints`, `[1,1,1]`) to multiply a `16x16x32` map in the keypoint graph. The FP16
binary vector path now broadcasts that scalar across every 32-channel block. `HX_ZEROINSERT`
also copies blocked channel vectors onto the sparse output grid, preserving its zeros and odd-width
pad columns. A second blocked gather packs 3x3 stride-one convolution windows with whole
32-channel input blocks, including the pad-two border expansion at keypoint op 68. Unsupported
shapes retain the generic im2col gather. Unsupported binary broadcasts and zero-insert layouts
retain the scalar path. On
the single-thread Hexagon ISS, the Mercury scalar multiply fell from 2,298,294 to 29,556
pcycles, and its `10x10x63` to `21x21x63` zero insert from 3,575,466 to 57,615 pcycles.
A pad-two `8x8x32` to `10x10x32` convolution fixture fell from 383,451 to 97,491
pcycles with the same FP16 result.
These pcycles count instructions rather than predicting Frame latency.

Ordinary FP16 external input packing now splits rows across the four DSP workers. Mercury's
one-channel image is represented with a 32-channel stride, so import has a large padded footprint.
The Frame profile placed 6.58 million detection cycles outside operator timing, which includes
import and export; the precise share due to import has not been measured on device. Parallel
packing is a candidate for reducing that interval, pending a Frame measurement. The HXFG and
`.pte` files do not need re-export for these engine changes.

The serial whole-graph ISS harness measured 23,700,072 to 16,853,346 pcycles for keypoint
and 7,522,407 to 7,475,226 for detection, including import but excluding constant staging
and output export. The changed keypoint graph produced bit-identical outputs to the prior
simulator engine on all four outputs. The simulator's absolute Mercury outputs differ from the
saved Frame outputs even on the prior engine, so this is a relative regression check, not a
Frame accuracy result. The saved prior Frame keypoint heatmap matched the ONNX Runtime argmax
for 20 of 21 joints; the remaining joint needs device validation with this engine.
