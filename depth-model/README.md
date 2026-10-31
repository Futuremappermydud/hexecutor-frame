# Extracted Legacy headset stereo depth model

The two Model stereo models have been extracted into **valid Qualcomm QNN HTP
context binaries**. Qualcomm's `qnn-context-binary-utility` from QAIRT 2.50 accepts
both files and exports their graph I/O metadata.

**Both original contexts now execute, unmodified, in QAIRT 2.50's x86_64 HTP
simulator** (stock `qnn-net-run` + `libQnnHtp.so`, under Docker on the Mac). No
custom op package is needed. Runs are deterministic. Synthetic stereo probes
show the float output tracking horizontal disparity in pixels. All 76
convolutions per model now have a bit-exact integer reference, validated
against the SDK compiler and simulator. See
[Simulator execution and reference arithmetic](#simulator-execution-and-reference-arithmetic).
**Oct 30: both original DSP-side ExecuTorch programs now run on the Model's
signed vendor QNN runtime, at ~9.7 ms warm per RPC call. Hardware differs from
the simulator; all 54 cases were measured. Per-op wait/overlap profiles and
reproduction are in [the executor log](frame/RESIDENT_EXECUTOR_WIP.md).
Both recovered graphs also run on Frame through the hybrid ARM/HVX backend;
see [the updated port](frame/DIRECT.md). The experimental resident DSP port
now runs in ~1.10–1.13 s warm (was ~1.70 s), exact across all 54 cases; it
remains slower than the ~115 ms exact hybrid. Its memory/layout bottlenecks
and separate opt-in build are recorded in the executor log.**

The continuation also decoded every serialized operation/tensor record in both
models and exported all stored constants. The primary has 5,870 operations and
5,336 tensor definitions, including **76 convolution weight arrays in HWIO
order** (75 convolutions and one transposed convolution). See the recovery
section below. This is the compiled, tiled graph. A standalone memory-level NumPy reference
now reproduces both models' native simulator outputs bit for bit; see the
[full-graph reference](#full-graph-reference-status-and-handoff-oct-25-2026-continuation).
An untiled ONNX or training-model reconstruction has not been produced.

The updated Frame investigation establishes custom scalar/HVX callbacks. It
does not establish QNN loading. The extracted context needs a QNN HTP runtime;
it is not a freestanding function that can be passed to Valve's callback.

## Artifacts

| File | Purpose |
| --- | --- |
| `models/stereo_model.ptl` | Preserved original, active model from Model |
| `models/stereo_model_2.ptl` | Preserved original alternate model |
| `models/primary/context.bin` | Raw, SDK-validated QNN context, 7,668,584 bytes |
| `models/alternate/context.bin` | Raw, SDK-validated QNN context, 7,660,616 bytes |
| `models/{primary,alternate}/manifest.json` | Hashes, layers, ExecuTorch methods and I/O ordering |
| `models/{primary,alternate}/outer.pte` | ARM-side ExecuTorch program |
| `models/{primary,alternate}/dsp.pte` | DSP-side ExecuTorch program |
| `models/{primary,alternate}/qnn_delegate.fb` | Bolt QNN wrapper; do not pass this to QNN |
| `metadata/{primary,alternate}.qnn.json` | Unmodified output from Qualcomm's metadata utility |
| `metadata/{primary,alternate}.htp-strings.json` | Compiler options and kernel type strings |
| `metadata/v75-runtime-imports.json` | Local SDK V75 library's undefined symbols |
| `recovered/primary/graph.json.gz` | Decoded compiled operation/tensor records, compressed |
| `recovered/primary/operations.csv` | Operation list with original activation names |
| `recovered/primary/constants/convolution_weights.npz` | 76 named int8 HWIO convolution weight arrays |
| `recovered/primary/constants/constants.json` | Constant tensor types, shapes, quantization, files and hashes |
| `recovered/primary/constants/layers.csv` | Per-layer weights, compiled parameters and tile counts |
| `recovered/primary/verification.json` | Structural and SDK machine-code address checks |
| `recovered/{primary,alternate}/conv_oracle.json` | Per-conv SDK `Fi` regeneration, inverted biases and exact requant results |
| `sim/Dockerfile`, `sim/qnn_op_harness.cpp` | x86 HTP simulator runtime image and one-op QNN graph builder |
| `tools/run_qnn_sim.py` | Run an extracted context in Qualcomm's x86 HTP simulator |
| `tools/stereo_probes.py` | Synthetic known-shift stereo probes and output summaries |
| `tools/conv_oracle.py` | Per-layer compiler/simulator validation of convolution arithmetic |
| `tools/htp_emu.py`, `tools/run_htp_emu.py` | Standalone memory-level NumPy inference for both graphs |
| `tools/verify_htp_emu.py` | Full-run comparison against all recorded probe hashes |
| `frame/`, `frame/dist/depth-model-frame.zip` | Hybrid Frame runner: HVX convolution, ARM graph operations, packaged models |
| `recovered/{primary,alternate}/execution_order.json`, `kernel_calls.json` | Context-bound execution plans and compiled kernel arguments |
| `recovered/{primary,alternate}/emulator_*.json` | Exact per-op and full-run validation results |
| `reference/` | Probe output hashes and summaries from the simulator |

Equivalent artifacts for the alternate model are in `recovered/alternate/`.

Primary context SHA-256:
`2686e65717b81e2befcf75ecd4d610b77b748200c6f3ca3e68dc90b3cdbb9e93`.

Alternate context SHA-256:
`3a3df18364719208769d6acaa081a0bed07e90a7227be195e4a5f1667bfd8f57`.

## The missing extraction layer

The full primary nesting is:

```text
TH1 header, 55 bytes
  etz0 wrapper + deflate
    outer ET12 program: HexagonRpcBackend
      nested ET12 program: QnnBackend
        Bolt flatbuffer wrapper
          field 0: graph name
          field 1: input tensor IDs [1, 2, 3, 4]
          field 2: input ranks [4, 4, 4, 4]
          field 3: output tensor IDs [5, 6]
          field 4: output ranks [4, 4]
          field 5: raw QNN context byte vector at offset 0x80
```

The older scratch file `stereo_qnn_payload.bin` was the **whole Bolt wrapper**.
Qualcomm's utility rejects that file and accepts the field-5 byte vector.
The context ends before the wrapper's trailing name and tensor-ID vectors, so
removing only the first 128 bytes is insufficient for an exact extraction.

The extractor reads the actual vector length, verifies sizes and nesting, and
writes each layer unchanged. It requires Python 3 and no ML packages:

```sh
python3 depth-model/tools/extract_depth.py \
  depth-model/models/stereo_model.ptl depth-model/models/primary
python3 depth-model/tools/extract_depth.py \
  depth-model/models/stereo_model_2.ptl depth-model/models/alternate
```

## Verified tensor contract

Both contexts report graph name `zelda_qvga_boltnn_delegate - delegate_0`, build
`v2.19.0.240124133650_81096`, core API 2.13.0, backend API 5.19.0,
and 1,769,472 bytes of spill/fill memory. Tensor dimensions below are exact;
their interpretation as image axes and the camera rotation remain unverified.

| QNN input | Native type | Dimensions | Bytes |
| --- | --- | --- | --- |
| `x` (ID 1) | unsigned fixed-point 8 | `[1,1,320,256]` | 81,920 |
| `y` (ID 2) | unsigned fixed-point 8 | `[1,1,320,256]` | 81,920 |
| `hidden` (ID 3) | float32 | `[1,2,320,256]` | 655,360 |
| `t` (ID 4) | float32 | `[1,2,4,4]` | 128 |

`x` and `y` have scale `0.003921568859368563` and zero point 0.
With native input files, provide raw uint8 bytes; the quantization describes
their real-valued interpretation. `hidden` strongly suggests temporal state.
Engine strings independently name `temporal_hidden_state`,
`resetTemporalHiddenStateEveryNthFrame`, and `RnnMotionCompensation`.
The exact state initialization/update, meaning of `t`, camera order,
rectification, normalization beyond tensor quantization, and geometric units
still need recovery or a reference capture.

| QNN output position | Name | Native type | Dimensions |
| --- | --- | --- | --- |
| 0 / ID 5 | `boltnn_permute_from_boltnn_tensor_81` | unsigned fixed-point 8 | `[1,2,320,256]` |
| 1 / ID 6 | `quantized_decomposed_dequantize_per_tensor_default_333` | float32 | `[1,2,320,256]` |

The uint8 output dequantizes as `(q - 136) * 0.05533868819475174`.
The **outer ExecuTorch method returns these outputs in reverse order**:
float32 first, uint8 second. Simulator probes with synthetic shifted textures
show the float output tracking **horizontal disparity in pixels**, with `x` as
left and `y` as right. The uint8 head behaves like an uncertainty score
(see below). Neither is yet established as metric depth or calibrated
confidence on real camera input.
The preliminary scratch I/O JSON's semantic guesses are not established facts.

## V75 compatibility and the updated callback findings

Snapdragon 8 Gen 3 is SM8650, QNN SoC ID 57, Hexagon V75. The Model uses a V69
runtime, but the context's compiler-options string records `min_arch=68`,
`vtcm_mb=8`, and `soc_type=UNKNOWN_SDM`. These strings do **not** prove V75
compatibility or incompatibility. Only backend deserialization and execution
can settle that. The 8 MB VTCM request can also affect whether it loads.

No explicit `BoltnnOpPackage` string appears in either extracted context. The
`boltnn_*` activation names alone do not establish a custom-op dependency.
Try stock QNN first on a working target; investigate package/version errors if
they actually occur. Metadata acceptance by QAIRT 2.50 proves metadata
readability, not compatibility of every serialized HTP kernel with that SDK.
**Update:** QAIRT 2.50's stock x86 HTP simulator deserializes and executes
both contexts with no op package, which settles the custom-op question for the
stock 2.50 HTP kernel set. Loading on V75 hardware remains untested.

The installed SDK's `libQnnHtpV75.so` imports 46 `qurt_*`/`HAP_*` functions,
including thread creation, mutexes/conditions, HVX locks, memory mapping and
power control. Loading that library through Valve's callback would require
an ELF loader plus the required OS/runtime services, or a functioning QuRT
path. A successful HVX matmul does not supply those services.

For a custom callback inference backend, this context is a reverse-engineering
input, not an executable payload. The continuation recovered its serialized
tensor references, shapes, quantization, storage references, constant pool and
HWIO convolution weights. Reproducing the optimized operator arithmetic,
buffer aliasing and scheduling is still necessary before adapting kernels.
No ONNX conversion or numerical equivalence is claimed.

## Compiled graph and weight recovery

The HTP pickle header is at context offset `0x3368` in the primary. Its public
magic `0x7309f72b` and tagged header lead to operator and tensor-class
dictionaries. Numbered operation markers use XOR constant `0x1303ee71`.
These constants and tensor format fields are documented by the installed SDK
headers under `include/QNN/HTP/core/`.

The decoder handles shared shapes/interfaces, compressed dimensions, scalar
and shape tensors, direct and compressed indirect storage references,
variadic operations, framework data vectors, and the final fixup terminator.
Every operation record must be consumed exactly; unknown structures raise an
error. All decoded input references resolve to earlier tensor definitions.
This validates the serialized reference structure, while framework buffer
aliases and checkpoint semantics still need interpretation for execution.

| Recovered item | Primary | Alternate |
| --- | --- | --- |
| Serialized operations | 5,870 | 5,848 |
| Tensor definitions | 5,336 | 5,343 |
| Serialized kernel types | 80 | 80 |
| Original activation debug names | 199 | 199 |
| Stored constant tensors | 230 | 231 |
| Stored convolution weight tensors | 76 | 76 |
| Logical int8 convolution weight bytes | 7,096,832 | 7,096,832 |

The primary constant extent descriptor is at `0x6e368`. It identifies constant
pool 5 at context offset `0x6f368`, with length **7,209,216 bytes**. The export
preserves this entire pool and writes each stored constant as `.npy`.

The `xwb` and `xpb` signed-byte weight tensors use different address layouts.
Their SDK type names are `SWeights` and `pkSWeights`. Following the SDK's RTTI,
relocations and vtables reaches `element_addr`, which calls ARM64 address
routines at `0x136d180` and `0x136d4b4` in the inspected QAIRT 2.50
`libQnnHtpPrepare.so`. Translating those routines yields ordinary HWIO arrays.
The verification tool independently interprets the original SDK ARM64
instructions and compares 6,080 boundary/random address cases per model.
Both models pass. This checks the byte permutation, not network predictions.

Every one of the **80 serialized kernel type names** also occurs verbatim in
the installed stock V75 `libQnnHtpV75.so`. No serialized kernel names Meta's
`BoltnnOpPackage`. This is stronger evidence for trying stock QNN than the
earlier string inventory, but it does not establish ABI compatibility or
successful graph deserialization.

Run the recovery locally:

```sh
python3 depth-model/tools/htp_graph.py \
  depth-model/models/primary/context.bin depth-model/recovered/primary
python3 depth-model/tools/export_constants.py \
  depth-model/models/primary/context.bin \
  depth-model/recovered/primary/graph.json \
  depth-model/recovered/primary/constants
python3 depth-model/tools/verify_recovery.py \
  depth-model/recovered/primary depth-model/metadata/primary.qnn.json \
  --sdk-prepare "$QNN_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2/libQnnHtpPrepare.so"
```

Only the optional SDK instruction check needs Capstone. The extraction,
decoder and array exporter otherwise use Python's standard library.
The instruction check is guarded by the inspected SDK library's SHA-256.

With NumPy installed, the named weight archive is directly readable:

```python
import numpy as np
weights = np.load("depth-model/recovered/primary/constants/convolution_weights.npz")
w0 = weights["boltnn_quantized_conv2d_nhwc_tensor"]
assert w0.shape == (1, 1, 12, 32)
assert w0.dtype == np.int8
```

The first compiled layer uses im2col packing, so its stored `1x1x12x32` array
does not mean the original camera convolution was trained as a 1x1 filter.
Likewise, `fi` constants are **compiled bias/requantization parameters**, not
ordinary float biases. The archive preserves compiled int8 weights and their
quantization metadata; it is not a recovered floating-point training model.

To run through the Frame callback, the next step is a reference implementation
of these compiled operators, starting with the 8-bit/16-bit convolution and
its packed requantization parameters. It must also model shared buffers and
checkpoint ordering, then be checked against reference QNN outputs before
claiming equivalent depth. Native camera rectification, `hidden` state, `t`
transform conventions and output units remain separate integration work.

## Simulator execution and reference arithmetic

### Running the original contexts locally

QAIRT ships an x86_64 HTP backend that runs Qualcomm's own graph executor and
HTP kernels on the host CPU. It restores the extracted contexts directly:

```sh
python3 depth-model/tools/prepare_inputs.py /tmp/qd-in --smoke
python3 depth-model/tools/run_qnn_sim.py \
  --sdk "$QNN_SDK_ROOT" --context depth-model/models/primary/context.bin \
  --input-list /tmp/qd-in/input_list.txt --output /tmp/qd-out
```

`run_qnn_sim.py` builds `depth-model/sim/Dockerfile` (Ubuntu 22.04 amd64 plus
libc++), mounts the SDK, and runs stock `qnn-net-run --retrieve_context`. Each
input-list line is one inference, about 8.5 s under amd64 emulation on Apple
Silicon. Observed:

- Both primary and alternate execute with status 0. The only warning is the
  expected `Pickle and software version mismatch v2.19.0 ... vs v2.50.0`.
- Reruns are bit-identical. Smoke-input output SHA-256s (primary):
  uint8 `dfacee19...6ad3`, float `a978ee82...b32b`.
- With the smoke input, the alternate's uint8 output is bit-identical to the
  primary's; the float outputs differ.

This is a **simulator** result. It shows the stock 2.50 HTP runtime accepts
both contexts and gives a deterministic reference for this context under
these kernels. It does not show V75 hardware loading, real-time performance,
or bit equality with the Model's V69 runtime.

### What the outputs mean (synthetic probes)

`tools/stereo_probes.py make` crops two windows of one procedural texture with
a known shift. Median interior output, primary, zero hidden, identity `t`:

| Case | float ch0 | float ch1 | uint8 dequantized ch0 |
| --- | --- | --- | --- |
| `y` = `x` shifted so features move **left by 2/4/8/16/32 px** | 2.00 / 3.99 / 7.97 / 15.63 / 31.90 | 2.00 / 3.96 / 7.89 / 15.74 / 32.32 | −3.3 to −5.0 |
| Same image (0 px) | 6.65 | 6.52 | −4.1 |
| Features moved right, or vertical shifts | 49–71 | 49–62 | −1.1 to −4.0, high variance |

- **Float output ≈ horizontal disparity in pixels** along the 256-wide axis.
  Its producer is a 16-bit transposed conv/depth-to-space with scale
  0.00125 (range 0–81.9), then `Dequantize`.
- `x` is the left image and `y` the right: a point at column `c` in `x` is
  matched at column `c − d` in `y`.
- The uint8 head is lower for valid matches and higher, with more variance, for
  impossible ones. It behaves like an uncertainty score, but that label is
  unverified.
- Zero disparity is not reproduced (≈6.6). The likely cause is the training
  range or the synthetic input. Real captures are needed.

These probes are synthetic textures, not camera images; they establish axis,
sign and scale, not real-scene accuracy. Camera rotation, rectification and
metric conversion (baseline × focal / disparity) remain unverified.

### How `hidden` and `t` enter the graph

Tracing the compiled graph from the two `*InputSlice` input bindings:

- `t` only reaches `Quantize → Abs → Mean` over all 32 values,
  then `s = sign(0.00075 − mean|t|)`, then `g = (0.5·s + 0.5)·0.3 + 0.5`.
  So **`g ≈ 0.8` for zero/small `t` and `g ≈ 0.5` above the mean-absolute
  threshold**, and `g` scales the hidden branch. This corrects the earlier
  reversed gate description; the scalar path is now checked exactly. The matrix entries themselves never reach the
  network. Any motion compensation (Engine string `RnnMotionCompensation`)
  must warp `hidden` outside this graph.
- `hidden` is quantized to uint16, scale 0.0014038, range 0–92 (negatives clip
  to 0). It passes through a 4×4 stride-4 conv (`conv2d_nhwc_tensor_72`),
  is multiplied by `g`, and is blended through a learned weight
  (`mul_op_crt`) near the final 4× upsampling.
- The float output (0–81.9) fits `hidden`'s range. It is structurally
  consistent with feeding the warped previous output back as `hidden`. The
  warp and the meaning of channel 1 are unverified.
- On these synthetic inputs, `hidden` changed medians by at most ~2 px, and
  only in textureless regions.

### Bit-exact convolution arithmetic

`tools/conv_oracle.py` turns each stored convolution into a one-operator QNN
`Conv2d`, compiles it with the SDK's HTP prepare, and runs it in the simulator
through `sim/qnn_op_harness.cpp`, a small QNN C API graph builder. The harness
binary is built once inside the Docker image:

```sh
docker run --rm --platform linux/amd64 -v "$QNN_SDK_ROOT":/sdk:ro \
  -v "$PWD/depth-model/sim":/src depth-model-qnn-sim \
  g++ -O1 -std=c++17 -I/sdk/include/QNN -o /src/build/qnn_op_harness /src/qnn_op_harness.cpp -ldl
python3 depth-model/tools/conv_oracle.py depth-model/recovered/primary \
  --sdk "$QNN_SDK_ROOT" --harness depth-model/sim/build/qnn_op_harness --work /tmp/qd-oracle
```

Results for **all 76 convolutions in both models**
(`recovered/{primary,alternate}/conv_oracle.json`):

1. **The SDK regenerates each layer's compiled `Fi` tensor bit-identically**
   from the recovered input/weight/output scales, HWIO weights and an int32
   bias. All 12 packed `Xpb` weight tensors per model also regenerate
   byte-identically, as do 8 of the 64 `Xwb` tensors (the 1×1 and 16-bit
   layers). The other 56 `Xwb` byte searches miss because the exported 3×3
   weights have their kernel x-axis reversed relative to the convolution
   kernel. The later full-graph reference corrects that permutation when
   executing them. The archive still preserves the stored order. This
   confirms the recovered scales, biases and im2col weight order.
2. **`Fi` layout:** one 64-word row per group of 32 output channels,
   `[multiplier ×32, bias ×32]`. The bias half is what the kernel uses. For
   8-bit activations, `Fi_bias = bias + round(zp_out / M)`. For 16-bit
   activations, `Fi_bias = floor(bias / 256) + fold(zp_out)`; there the true
   model bias is only known to within 256 accumulator units, but the kernel
   sees only `Fi`.
3. **Closed-form requantization reproduces every simulator output exactly:**

   ```text
   acc = Σ (x − zp_in) · w + Fi_bias · (256 if 16-bit activations else 1)
   M   = s_in · s_w / s_out
   p   = 8 for 16-bit activations, else floor(−log2 M) − 3
   y   = clip(((acc >> p) · fp16(M · 2^p) rounded half-up), 0, qmax)
   ```

   `>>` is an arithmetic floor shift. The scale behaves as an fp16 value; a
   textbook `round(acc · M)` matches only 95–99.7% of 8-bit outputs. This is
   implemented as `conv_oracle.requantize`.

Scope: verified on random inputs, with padding and tile sizes chosen by the
test graph. Real padding comes from the compiled tile geometry and is not yet
extracted per layer. The other compiled operator types (`linearclip_simple_rq`,
`Add.tcm`, `mul_op*`, `addsub_op*`, `ResizeNear`, `Decimate`, `d2s`, the
dynamic-weight cost-volume convs) are not yet referenced. They can be
calibrated with the same harness.

### Remaining work for a full-graph reference

Tensor references alone are not a complete DAG. `@DummyOp*` nodes declare views
(block tables over 2 KB crouton blocks written by other tiles, e.g. conv halos).
`@Spill`/`@Fill` copy VTCM↔DDR, with operands in their framework vectors:
`[0, 1, n, ddr_offset, (vtcm_offset | pool), bytes, ...]`, where pool 1 is
the 8 MB VTCM and the DDR area is the 1,769,472-byte spill region. `@BlockZap`
zeroes listed blocks. Two workable routes:

- a **memory-level emulator**: replay ops in sequence over the VTCM/DDR
  pools, which gets topology from addresses; or
- a **logical graph**: collapse tiles to the 199 source ops, using block-address
  overlap for Dummy/Spill/Fill edges, then implement each op with the
  calibrated integer rules.

Validate either one against `run_qnn_sim.py` outputs end to end. Bit equality
is the target, because every operator's rounding is now observable.

## Runner for an already working QNN environment

Run the metadata utility on a supported Linux/Android host:

```sh
LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/$QNN_TARGET" \
  "$QNN_SDK_ROOT/bin/$QNN_TARGET/qnn-context-binary-utility" \
  --context_binary depth-model/models/primary/context.bin \
  --json_file primary.qnn.json
```

Prepare exact native input tensors captured or produced by your pipeline:

```sh
python3 depth-model/tools/prepare_inputs.py /tmp/depth-model-inputs \
  --x left.raw --y right.raw --hidden hidden.raw --t transforms.raw
```

`--smoke` generates synthetic tensors for checking runtime plumbing only.
Zero hidden state and identity matrices are placeholders, not recovered model
semantics, and successful execution would not establish meaningful depth.

On an Android SM8650 with functioning vendor QuRT/FastRPC and a matching SDK
deployment (the Mac cannot execute these ELF tools):

```sh
python3 depth-model/tools/run_qnn.py \
  --sdk "$QNN_SDK_ROOT" --target aarch64-android \
  --context depth-model/models/primary/context.bin \
  --input-list /tmp/depth-model-inputs/input_list.txt \
  --dsp-libs "$QNN_SDK_ROOT/lib/hexagon-v75/unsigned" \
  --output /tmp/depth-model-output
```

For supported Linux targets, select the SDK's corresponding target directory.
The runner sets SoC 57/V75, native input/output types, and DSP library search
paths. It records its command and log. Optional `--op-package` passes an
explicit QNN registration string, if a dependency is established. `--dry-run`
only writes configuration and prints the command. The runner has been checked
locally in dry-run mode; NPU execution remains untested.

## Checks performed

- Re-extracted both original files, checked wrapper sizes/ranks and hashed all layers.
- Primary decompression exactly matches the original analysis's `.dec` artifact.
- Qualcomm QAIRT 2.50 metadata utility: whole Bolt wrapper rejected (exit 1),
  both raw contexts accepted (exit 0), with matching tensor IDs/types/shapes.
- Checked native input sizes, synthetic input generation, and runner configuration.
- Read the revised Frame callback investigation. No callbacks, service changes,
  firmware changes or NPU inference were performed for this extraction.
- Continuation (local only; the Frame was not touched): executed both contexts
  in the QAIRT 2.50 x86 HTP simulator, checked bit-identical reruns, ran the
  synthetic stereo probes, and validated all 76 convolutions per model against
  SDK compilation (`Fi` bytes) and simulator execution (exact closed-form
  requantization).

References: [Qualcomm supported devices](https://docs.qualcomm.com/nav/home/QNN_general_overview.html?product=924033590759186372),
[QNN context restore API](https://docs.qualcomm.com/nav/home/typedef_QnnInterface_8h_1aa740a1b35494a0f4806ca4ad5919d1f3.html?product=924033590759186372).

## Full-graph reference: status and handoff (Oct 25, 2026, continuation)

`tools/htp_emu.py` now implements every executed kernel in both compiled graphs.
It runs independently of QNN, Docker and the SDK. NumPy and the recovered model
artifacts are sufficient. It executes the original tiled graph through shared
VTCM/DDR buffers; it is not an untiled ONNX reconstruction.

### Validation

- **Primary:** both native outputs match the simulator byte for byte for smoke,
  the seed-1234 random input below, and all 25 recorded stereo probes.
- **Alternate:** both outputs also match byte for byte for smoke, the same
  random input, and all 25 recorded stereo probes. That is **54 complete
  inferences / 108 output buffers** across the two models.
- All logical per-op outputs pass on a random trace for each model, including
  all **1,036 convolution tiles per model** and all 80 instances of every
  runtime-weight operation. Direct DDR outputs are checked as raw bytes.
- `calibrate_remaining.py` checks Abs on all 65,536 input values in both HVX
  lane positions (131,072 comparisons per parameter set), and mean on 4,096
  random/boundary vectors. All comparisons are exact.
- Whole-buffer `--strict` checks still detect padding differences, including
  the unspecified last convolution padding rows, unused lanes and deferred
  DMA writes at checkpoint waits. They do not affect the validated graph
  outputs. Logical per-op equality does not claim every padding byte matches.

Evidence is kept in `recovered/{primary,alternate}/emulator_*.json`, with
`recovered/primary/unary_calibration.json` for the direct helper checks. The
probe reference remains `reference/simulator_output_hashes.json`.

The reference arithmetic is calibrated against **QAIRT 2.50.0.260828's x86
simulator** for the compiled parameter sets these two graphs use. The original
QNN contexts have not been loaded on V69/V75 hardware. A new hybrid Frame
backend runs convolution dot products on V75 HVX and the other graph operations
on ARM; see [the Frame package](frame/README.md) for its separate hardware checks.

### Standalone inference

From `depth-model/`, using Python with NumPy (locally
`/opt/homebrew/bin/python3.13`):

```sh
python3 tools/prepare_inputs.py /tmp/qd-smoke --smoke
python3 tools/run_htp_emu.py recovered/primary \
  --inputs /tmp/qd-smoke --output /tmp/qd-emu-primary
python3 tools/run_htp_emu.py recovered/alternate \
  --inputs /tmp/qd-smoke --output /tmp/qd-emu-alternate
```

The runner writes native `.raw` outputs and a SHA-256 `report.json`. Optional
`--compare SIMULATOR/Result_0` compares every output element and returns nonzero
on a mismatch. Output names come from the model's existing QNN metadata.

`recovered/{primary,alternate}/execution_order.json` preserves a valid schedule
as **serialized node identities**, and `kernel_calls.json` preserves the
compiled kernel arguments. Both are checked against the context SHA-256.
The runner computes runtime scalars and weights from the supplied input files;
it does not load the trace memory snapshots. A trace directory may be
passed as the runner's optional second positional argument to override the
saved schedule.

```sh
python3 tools/stereo_probes.py make /tmp/qd-probes
python3 tools/verify_htp_emu.py recovered/primary /tmp/qd-probes \
  --report /tmp/qd-primary-probes.json
python3 tools/verify_htp_emu.py recovered/alternate /tmp/qd-probes \
  --report /tmp/qd-alternate-probes.json
```

The verifier reuses address maps between inferences and resets runtime memory
between cases. Observed execution is roughly 7 seconds per warm inference on
this Mac, with about 2 GB process memory. Convolution uses float64 BLAS only
for exact integer dot products: a checked bound keeps every product and
partial sum below `2**53`. Bias and requantization remain integer arithmetic.

### Simulator tracing and per-op checks

`sim/qnn_trace_hook.cpp` interposes the simulator's exported dispatchers and
HVX functions. It serializes run-list entries under a mutex and logs changed
2 KB VTCM blocks after each operation, plus kernel arguments. Original graph
outputs remain identical for the validated inputs.

```sh
SDK=/Users/owenelliott/Downloads/qairt/2.50.0.260828
docker run --rm --platform linux/amd64 -v "$PWD/sim":/src depth-model-qnn-sim \
  g++ -O2 -std=c++17 -shared -fPIC -o /src/build/qnn_trace_hook.so /src/qnn_trace_hook.cpp -ldl -lpthread
docker run --rm --platform linux/amd64 -v "$PWD/sim":/src depth-model-qnn-sim \
  g++ -O1 -std=c++17 -rdynamic -o /src/build/kcall /src/kcall.cpp -ldl
python3 tools/trace_qnn_sim.py --sdk "$SDK" --context models/primary/context.bin \
  --inputs IN --output TR --writes
# Rerun in the same directory to snapshot direct DDR outputs too:
python3 tools/trace_qnn_sim.py --sdk "$SDK" --context models/primary/context.bin \
  --inputs IN --output TR --writes --recovered recovered/primary
python3 tools/check_kernels.py recovered/primary TR --inputs IN --report /tmp/kernels.json
python3 tools/sim_trace.py recovered/primary TR
python3 tools/calibrate_remaining.py recovered/primary --sdk "$SDK" \
  --report /tmp/unary.json
```

Use `alternate` in the corresponding paths to trace/export the other model.
The DDR snapshot option needs the prior trace to prepare tensor-argument
slots. It records direct DDR output tensors in `tensors.bin`, including
runtime weights and scales. The checker compares them before restoring the
reference DDR bytes for the next operation, just as it restores VTCM before
each operation. It returns nonzero for missing kernels or mismatches.
`--skip-conv` skips HMX calculation during focused development; `--strict`
also compares changed VTCM blocks, including padding.

Run-list mapping is validated against the known function/type associations
in `sim/qairt_2_50_functions.json`. List 0 follows flags-0/2 nodes; list 1
follows flags-4/6 nodes. Flags-6 input slicing also has a main-list no-op.
The alternate has six additional executor-only no-ops after those stubs;
they have no serialized node and are omitted from its saved execution plan.
Thus the primary has 5,068 trace entries, while the alternate has 5,046 trace
entries and 5,040 saved entries. Execution interleaving can change; always
key comparisons and compiled arguments by node identity.

`kcall` can invoke exported kernels or internal helpers by library-relative
address on chosen data. Its `-rdynamic` build bypasses the simulator's HVX
worker-thread guard for direct scalar simulator calls. This change applies
only to the calibration executable.

### Recovered arithmetic and layouts

- Crouton8 chunks are h8 × w8 × d32; crouton16 chunks are
  `(h8, w//2%2, d32, w%2)`; crouton32 chunks are h8 × w2 × d32.
  `max_dimensions` includes padding, and `padding` is the pad-before offset.
  Block tables plus `slice_offset` implement aliases and halos.
- Stored stride-1 3×3 Xwb weights are x-reversed:
  `true[ky,kx] = stored[ky,kw-1-kx]`. The exported weight archive retains the
  stored order. Stride-2's nine stored slots represent taps
  `(0,2),(0,0),(2,2),(2,0),(0,1),(2,1),(1,2),(1,0),(1,1)`.
- Space-to-depth reads
  `out[h,w,32*(cb*4+2*dy+dx)+c] = in[2*h+dy,2*w-dx,32*cb+c]`.
- Fi stores 32 multiplier words followed by 32 biases for each output group.
  The multiplier's low half is fp16; `M=(1024+mant)/2**(34-exp)`.
  Convolution pre-shifts are `S-14` for u8 and 8 for u16.
- Resize uses a Q32 source step in parameter word 2, a fractional origin in
  word 4, and an integer origin in word 5. The models have different fractional
  origins. Treating word 4 as the step happens to work only for the primary.
- Abs applies Q18 scaling with distinct rounded/truncated intermediate lanes;
  in crouton16 these alternate by column. Mean's 32-element reduction uses
  `(sum + 33) >> 5`, saturated to u16, for these compiled parameters.
- Dequantization rounds exact halfway float products away from zero. This
  differs from NumPy's ties-to-even for some alternate-model values.
  Quantize retains the exact product of its float32 operands until integer
  rounding; a float32 intermediate changes random hidden-input cases.
- Runtime-weight conversion reduces per-channel extrema, computes a scale
  of at least 1, rounds its float mantissa to 11 fractional bits, converts
  the inverse scale to Q16 and packs signed weights. All intermediate scales,
  packed weights, and Fi construction match the direct DDR/VTCM snapshots.
  The implementation records the exact formulas.
- `@BlockZap` copies blocks and fills the encoded trailing rows; spill/fill
  copies byte ranges between VTCM and the spill area. Scalar operands that
  travel through DDR are read at runtime.

The other calibrated integer formulas remain in `htp_emu.py`. Identified
HVX multiply helpers include 392a5d0/5e0 (u16 × scalar), 392a620 (u16 × u16),
39277e0 (word add), 392b130 (word subtract), 392a2c0 (multiply even), 392a4e0
(multiply odd accumulate), 3927920 (saturating add), and 3927f30 (rounded,
saturating shift to unsigned halfwords).

Random input recipe: `rng=np.random.default_rng(1234)`; generate x then y as
uint8 `[320,256]`, hidden as float32 uniform(-5,95) `[2,320,256]`, and t as
float32 normal `[2,4,4]`, in that order. Primary output hashes are
`7038ddd966f6552664184d28c7d7e4ce228b7229533778e227878ed540253317` and
`31f6a27cc33892ece131692bdb70abea74fae20519cf0048c95fe072e2b5fb8e`.
The alternate uint8 hash is identical; its float hash is
`6ac85f5f33fdf8dfba88f53f7f8afa7ed9adc5931239d61c778569fc854bcc47`.

A hybrid HVX port and runnable Frame package are now in `frame/`. An untiled
portable graph, further DSP offload and performance optimization remain separate
tasks. Camera rectification, external
hidden-state warping and metric calibration remain separate integration work.

## Frame HVX package

`frame/dist/depth-model-frame.zip` is a runnable, checksummed package of both
models and the callback backend. It is installed on the Frame at
`~/frame-dsp/depth-model/depth-model-frame/`, with NumPy in
`~/frame-dsp/depth-model/.venv/`.

Both models have run on the Frame: smoke and seed-1234 random inputs match
both simulator outputs byte for byte, using 1,036 HVX convolution calls per
inference. The ARM CPU handles scheduling, layout/nonconvolution operations,
runtime weights and exact requantization. The normal measured runtime is
137–141 seconds per inference; it is not real time. Keep the headset awake.

```sh
ssh frame 'cd ~/frame-dsp/depth-model && .venv/bin/python depth-model-frame/run.py --smoke --output results/test'
```

See [frame/README.md](frame/README.md) for arbitrary inputs, packaging, local
and hardware validation, memory layout, and eye-client pause/restoration.
The callback, private arena and stock eye client were restored after testing;
firmware files and daemon configuration were not changed.

## Offline stereo-pair adapter (Oct 29, 2026)

`tools/model_depth.py` adds a reusable CPU session and an image/NPY/raw stereo-pair
CLI over the validated integer reference. It exports both float planes, the
native auxiliary head, dequantized auxiliary values, previews and a hash report.
`portable/dist/depth-model-cpu.zip` contains both models and the CPU code with
no Frame transport dependencies. See [usage and limitations](portable/README.md).
No Frame access was needed for this adaptation.

The 137–141 second hardware timing above belongs to the old Oct 25 hybrid build,
before later AXI clock/bandwidth work. It does not measure current full-model
performance after those fixes. The faster convolution path and HMX GEMM results
are separate kernel measurements, not an updated full-depth inference timing.

## Updated Frame direct-convolution port (Oct 29, 2026)

The optimized hybrid ARM/HMX path runs primary smoke inference in
**0.401–0.403 seconds warm** (0.530 seconds first inference), with exact
simulator output matches. Integer HMX handles 916 convolutions per inference;
HVX handles the remaining cases. Native C tensor access and data movement
replace the previous large NumPy address maps. This is about 2.5 FPS and
remains below real time. See [measurements and image-pair usage](frame/DIRECT.md)
and `frame/dist/depth-model-direct.zip`. Earlier timing sections describe
historical builds.


## Model runtime pipeline follow-up (Oct 29, 2026)

The model alone omits several implemented native stages. Offline investigation
of the saved retail `depth_engine.so` (SHA-256
`0e558058c08a6e22ea6e2fb1c8b7890be5a472b515076c84cf643c57f112babe`)
confirmed that model initialization conditionally constructs
`RnnMotionCompensation`. The native call at `0x9ea3e0` reaches the constructor
at `0x54cd60`; its setup logger names separate left/right history, pyramid level
and the Enoki path. Reset period and compensation quality are native settings.
The active values remain unverified without a connected Model.

Three GLSL sources are embedded verbatim in the same library. The recovered
code implements pose/camera reprojection of previous depth, inverse-depth merge,
segmentation and near-foreground bypass, and separable 15-tap Gaussian filtering
with powered means and blur-history freeze counters. This is a downstream
passthrough/depth-output filter, distinct from the model's hidden-state motion
compensation. Its hole filling and foreground suppression should not be copied
wholesale into a metric room mapper. The Gaussian kernels mask missing samples;
they do not contain a bilateral depth-difference weight.

`global_tsdf` and instantiated Vogon `SdfFusion` types provide static evidence
of a separate surface-reconstruction implementation. Strings for multi-view,
epipolar and left/right consistency counters are present too; their exact role
in the Zelda path has not been traced. Shader code presence and configuration
strings do not establish which stages are enabled in the current runtime.

The related CVPR 2023 paper [A Practical Stereo Depth System for Smart
Glasses](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_A_Practical_Stereo_Depth_System_for_Smart_Glasses_CVPR_2023_paper.pdf)
is useful for online rectification and calibration failure handling. It describes
Argos, a photo-effects system with right-reference normalized relative disparity.
It does not establish Zelda's scale, eye order, hidden-state contract or runtime
pipeline. Its four-parameter relative orientation/focal correction is a useful
Frame calibration diagnostic, rather than proof that our rig needs correction.

The current Frame integration zeros hidden state every frame and uses its own
census/support checks and point-voxel averaging. It does not reproduce Model's
motion-compensated history, output temporal filtering or TSDF fusion. Numerical
model parity is therefore insufficient to predict surface stability.

Verbatim shader sources, binary-bound extraction offsets, and native call-site
evidence are in `metadata/model-runtime/`. Reproduce the extraction from a saved
library without running it:

```sh
python tools/recover_model_pipeline.py /path/to/depth_engine.so --output metadata/model-runtime
```


### Live Model observations

On Oct 29, the connected Model's firmware and both model hashes matched the
saved artifacts. While Scene Setup was active, MrSystemService reported:

- Primary `stereo_model.ptl` open by the service and 256×320 dense depth output.
- `RnnMotionCompensation`: separate left/right history enabled, pyramid level 1,
  and Enoki enabled. This confirms the component was initialized in the test.
- Dense-client confidence enabled; Scene Reconstruction reported uncertainty
  maps present and `Confidence maps status: true, threshold: 89`. The threshold's
  units and relationship to the model's auxiliary output remain unverified.
- Scene Reconstruction actively fused frames at approximately 5 Hz and generated
  meshes around 1.3 Hz in the sampled interval. In one sample it reported 132
  depth frames received and 131 fused, with one connected component after cleanup.
- During active capture, service telemetry showed mean disparity compute about
  10.5 ms and core compute about 14.3 ms. These are device-reported stage timings,
  not a separately instrumented end-to-end Scene Setup latency.

The frameset was reported as grayscale; sensor and projector mode differed by
client. Exact preprocessing, exposure timing, hidden-state warping, the native
confidence transform and the camera-to-model convention still need tracing.
Model's calibration client reported no loaded factory/online calibration in this
particular service dump; that observation alone does not describe all calibration
sources used by the active camera pipeline.

The captured service dumps and logs are under `.work/model-live-20261029/`.
A temporary diagnostic Depth client and Frida server were removed afterward;
the user's active Scene Setup retained its own Depth client.
