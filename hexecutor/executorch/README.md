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
  host sources and `hexecutor/host/cpp/tensor_reference.cpp` by path. With `-DHEXECUTOR_ET_SHARED=ON` it also builds
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
mkdir -p hexecutor/.work/et-src
cd hexecutor/.work
git clone --depth 1 --branch v1.0.0 https://github.com/pytorch/executorch et-src/executorch
(cd et-src/executorch && git submodule update --init --depth 1 \
    third-party/flatbuffers third-party/json third-party/gflags third-party/flatcc \
    backends/xnnpack/third-party/FXdiv)

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

The direct device path requires all four firmware artifact paths. The broker and portable reference paths manage their inputs separately.

## Runtime behavior

- `init()` parses the blob, writes the HXFG image and constants into a private
  `mkdtemp` directory (mode 0700), loads the program with
  `hexecutor::load_tensor_program`, and checks it against the blob contract:
  count, float32 type, and shape (NHWC for layout 1, exact dims for layout 0).
- With the installed broker socket (or an explicit `HEXECUTORD_SOCKET`), `init()`
  registers the embedded graph with `hexecutord`; `execute()` submits through
  the broker. The daemon currently has eight registration slots per lifetime.
- Direct device mode: every delegate handle in the process shares one multi-program
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
- `HEXECUTOR_ET_REFERENCE=1 hexecutor_et_selftest samples/hand-tracking/.work/single45.hxfg samples/hand-tracking/.work/landmark.bin single45-input.f32 single45-oracle.f32`: outputs match the stored oracle exactly (max abs diff 0 over 9408 values). NCHW input equals NHWC input, and a wrong shape returns `InvalidArgument`.
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

- Direct mode shares a single session per process. Broker mode registers programs with the daemon. Verify slot growth and recovery on the target device with the actual group of delegates. [Broker lifecycle](../../docs/hexecutor/broker.md)
- Float32 I/O only. Batch size and graph shape are fixed by the blob.
- The CPU reference copies the image and constants from disk on every run, so
  it is for correctness checks only.
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

See the model guide for the Mercury HXFG reference and simulator comparisons. [Mercury validation](../../docs/model-development/mercury-hand-tracking.md#hxfg--executorch-simulator-evidence)
