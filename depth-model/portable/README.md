# Model stereo: offline CPU adapter

Both recovered Model stereo models can now be used through a reusable Python
session or a command line, without QNN, Docker, a headset or DSP services.
This is a usable experimental disparity reference, not a calibrated camera
integration or a real-time NPU implementation. The tiled integer graph is
preserved; this is not an ONNX conversion.

## Run from the original project

```sh
cd /Users/owenelliott/Documents/projects/spatial-stuff/depth-model
.venv-cpu/bin/python tools/model_depth.py \
  --left left.png --right right.png --output results/my-pair
```

The local `.venv-cpu` has NumPy and Pillow available. For a new environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r portable/requirements.txt
.venv/bin/python tools/model_depth.py --left left.png --right right.png --output results/my-pair
```

## Run the portable archive

Unzip `portable/dist/depth-model-cpu.zip` on a machine with Python 3.10+:

```sh
cd depth-model-cpu
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py --left left.png --right right.png --output results/my-pair
```

`--model alternate` selects the second model. Inputs can be image files,
uint8 `.npy` arrays of shape `[320,256]`, or exactly 81,920 bytes of native
uint8 `.raw`. Images are converted to grayscale. Model coordinates are
**256 pixels wide and 320 high**. Incorrect image dimensions are rejected;
`--resize` explicitly permits resizing image files. Resize does not rectify
stereo images or resolve camera orientation, and does not change `.npy` or
raw tensor dimensions. Caller supplies an already rectified pair in the right
model orientation; do not infer a rotation from the unusual dimensions.

By default, each call starts with zero hidden state and two identity matrices.
These are experimental single-pair placeholders. Supply `--hidden state.npy`
(float32 `[1,2,320,256]`) and `--transforms matrices.npy` (float32 `[1,2,4,4]`)
if the correct state and motion conventions are known. Output is never fed
back implicitly. External temporal warping and state semantics remain unresolved.

## Outputs

- `outputs.npz`, containing all three arrays below.
- `float_output.npy`: float32 `[1,2,320,256]`. Synthetic tests show disparity-like
  behavior in pixels. Both output planes are retained without assigning eye semantics.
- `auxiliary_u8.npy`: the native uint8 head, same shape.
- `auxiliary_dequantized.npy`: `(q - 136) * 0.05533868819475174`.
  This head may describe uncertainty; it is not established calibrated confidence.
- `float_plane_0.pgm` and `float_plane_1.pgm`: grayscale previews stretched
  independently to their 2nd–98th percentile ranges, recorded in the report.
  These previews are for inspection; use the arrays for numeric work.
- `report.json`: context identity, input paths, placeholder choices, elapsed
  inference time, output shapes/types and SHA-256 hashes over native array bytes.

No metric-depth conversion is applied. Camera intrinsics, baseline, camera order
and real-image geometric validity must be established first. Raw output planes
are in model coordinates. This implementation uses approximately 2 GB of memory;
first inference builds address caches and takes longer than subsequent calls.

## Python API

From the project root (or the unpacked archive root):

```python
import sys
sys.path.insert(0, 'tools')
import numpy as np
from model_depth import StereoModel

model = StereoModel('primary')  # reuse the address caches for subsequent pairs
left = np.load('left.npy', allow_pickle=False)    # uint8 [320,256]
right = np.load('right.npy', allow_pickle=False)
result = model.infer(left, right)
planes = result['float_output'][0]                # float32 [2,320,256]
```

Inputs are checked for exact shape, dtype and finite values. Outputs are copied
so subsequent calls do not mutate earlier results. A session is not thread safe.

## Validation and rebuilding

```sh
.venv-cpu/bin/python tools/test_cpu_adapter.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv-cpu/bin/python tools/test_cpu_adapter.py --integration
.venv-cpu/bin/python tools/build_cpu_package.py
```

The integration check compares both native outputs against existing simulator
hashes for both models: smoke, an 8-pixel synthetic stereo shift, arbitrary
seed-1234 state/transforms and a final smoke call to check reset behavior.
It also reloads exported NPZ arrays and checks exact equality. Results are in
`portable/validation.json` (`validation.json` in the archive). Unit checks cover
bad inputs, image/raw/NPY loading, dimension rejection and explicit resizing.
The archive includes checksums for every payload file in `manifest.json` and
contains no Frame transport, payload, SSH command or hardware dependencies.

## Hardware performance context

The old 137–141 second Frame result is from the Oct 25 hybrid HVX/ARM build,
before subsequent AXI clock/bandwidth work. On Oct 29 the repaired direct-HVX
path passed both models on hardware: 23–24 s first inference and 11.5–11.9 s
warm, with exact simulator output hashes. That path also changes the kernel
and command protocol; these timings do not isolate the clock fix alone.
Standalone HMX GEMM works; the quantized stereo graph has not been ported to it.
This archive remains CPU-only. The hardware port is distributed separately as
`frame/dist/depth-model-direct.zip` in the source project.
