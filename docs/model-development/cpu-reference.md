# Offline CPU depth reference

The CPU adapter runs the recovered compiled graph without Frame transport or device access. It is for numerical comparison, output inspection, and offline development. It is not the removed hybrid Python/DSP backend.

From the monorepo root, after separately provisioning recovered graphs/constants and installing NumPy:

```sh
python3 samples/depth-model/tools/model_depth.py \
  --model primary --left LEFT.npy --right RIGHT.npy --output OUTPUT_DIR
```

Use `--help` for raw input, explicit resizing, hidden-state, and transform options. Inputs are uint8 320 × 256 in model coordinates. The tool does not rectify a headset image, recover its camera calibration, or guarantee a physical depth scale. Requesting resize does not make an arbitrary pair geometrically valid.

`StereoModel` is a reusable mutable session. It copies returned arrays so a later inference does not overwrite an earlier result; do not use one session concurrently. Explicit inputs allow float32 hidden state `[1,2,320,256]` and transforms `[1,2,4,4]`. Omitted state is zero; omitted transforms in this CPU adapter are identity. Match the live bridge's zero-transform behavior explicitly when comparing its no-history mode.

Outputs include both float planes, the native auxiliary head, dequantized auxiliary values, previews, and hashes/provenance. The float output is treated as disparity by the current runtime. Preview normalization and auxiliary dequantization are visualization/reference operations, not confidence calibration.

`htp_emu.py` implements the memory-level numerical oracle; `run_htp_emu.py`, `sim_trace.py`, and related tools provide execution/graph inspection. `samples/depth-model/python/depth_model/reference/native_reference.py` is a CPU-only accelerated reference with native tensor access. Production DSP transport and kernel utilities remain separate.

`samples/depth-model/tools/build_cpu_package.py` can assemble a local portable package and reads this guide through the compatibility README path. Generated packages contain local artifacts and remain ignored. A source checkout alone is intentionally insufficient for full-model inference.
