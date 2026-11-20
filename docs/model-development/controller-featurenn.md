# XRService controller FeatureNN on Hexecutor

XRService's controller tracker runs a stereo keypoint network, FeatureNN, on the CDSP. This guide covers compiling the shipped CPU fallback model for the tensor backend and using the stock XRService adapter. [XRService bridge](../hexecutor/xrservice-broker-bridge.md#stereofeaturenn)

## Model

The model ships with SteamVR at `/opt/steamvr/drivers/cv/resources/minControllerFeatureNNModel/K2V3/` (SHA-256 `1dca8b6127063debd866203967275b5efb3e8c923794ceb49d7dec8fd4af703c` on the 2026-11-03 install). It is Valve's: keep copies and every derived file under the ignored `samples/controller-tracking/.work/`.

| Item | Value |
| --- | --- |
| Architecture | MobileNetV2 width 1.0, static onnxruntime quantization (u8 activations, per-channel int8 weights) |
| Input | `Image` `[1,2,128,128]` float, a stereo crop in 0–1; normalized by `(x − 0.449) / 0.226` before quantization |
| Body | 55 `QLinearConv`, 11 `QLinearAdd`, spatial mean |
| Head | Float Gemm 1280→552, reshaped `[2,92,3]` and split into `center`, `uncertainty` (`exp(2·min(x,10))`), `plus_x`, `plus_z`, each `[2,23,3]` |

## Lowering

The host performs the input normalization and the reshape/split/exp head. The DSP graph runs from the normalized input to the Gemm output:

```sh
python3 -m venv samples/controller-tracking/.work/venv
samples/controller-tracking/.work/venv/bin/pip install numpy onnx onnxruntime
samples/controller-tracking/.work/venv/bin/python hexecutor/offline/lower_onnx.py \
  samples/controller-tracking/.work/model/featurenn_k2v3.onnx samples/controller-tracking/.work/lowered/featurenn \
  --input /Div_output_0 --output-tensor /head/fc/Gemm_output_0 --pad-input-channels 3
```

The result has 68 operations, a 3.9 MB arena and 18.4 MB of constants. The two input channels are zero-padded to three so the compact RGB input path applies. The 552-wide Gemm is split into outputs of 352 and 200 channels, and the host joins them. Two layers get plain ReLU because their quantized range is not 6: the first convolution (0–62.1) and the final 1×1 (0–3.54). The u8 model saturates the latter at 3.54 and the float graph does not.

## Built-in XRService adapter

Register the result as `featurenn` and pass `--xrservice-featurenn featurenn` to `hexecutord`, or set `FRAME_FEATURENN` to the prefix when using `hexecutor/packaging/run_hexecutord.sh`. The adapter accepts the built-in binary's packed u8 crops over the stock `dsp_service` shared buffer, performs the normalization and output head layout described above, and runs inference as a tracking broker job. It does not patch or replace XRService. See [the XRService bridge guide](../hexecutor/xrservice-broker-bridge.md#stereofeaturenn) for the handshake, launcher example, and remaining device-validation gates.

## Checks (2026-11-09)

`samples/controller-tracking/tools/check_featurenn.py` compares the lowered graph with onnxruntime on the quantized model. Inputs were uniform noise and natural greyscale crops; no controller crops were available.

| Comparison | Worst absolute difference, raw head |
| --- | --- |
| Portable float reference vs onnxruntime u8 | 0.061–0.17 (center mean 0.009–0.018) |
| v75 simulator FP16 vs portable float reference | 0.003–0.005 |
| v75 simulator FP16 vs onnxruntime u8 | 0.055–0.071 |

The DSP kernels match the float lowering. The remaining gap is float versus u8 quantization. Whether it matters for tracking has to be measured on real controller crops. The simulator ran the FP16 plan at 2.79 M serial cycles: depthwise 2.06 M, convolution 0.54 M, add 0.11 M. Device time has not been measured.

To repeat the simulator check, write the input and reference files and run `hexecutor/tests/test_tensor_model_sim.c` (link with `-lhexagon`) on `hexagon-sim -mv75`, using the SDK and toolchain image recorded in the [MediaPipe validation instructions](hand-tracking.md#validation-and-performance).
