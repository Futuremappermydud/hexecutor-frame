# Learned depth integration

`ModelDepthPipeline` owns a persistent C++ worker for native DSP inference. It runs only in the background depth worker. A private inherited Unix socket carries two model images, motion/state inputs, and output arrays; camera DMA ownership remains in the C++ application.

## Backend and package selection

| Setting | Behavior |
| --- | --- |
| `FRAME_DEPTH_BACKEND=prepared-qnn` | Default, primary-only captured run list; the bridge pins a validated engine/bootstrap pair |
| `FRAME_DEPTH_BACKEND=prepared` | Exact prepared reference engine; primary and alternate packages |
| `FRAME_DEPTH_PACKAGE` | Installed package root containing the expected `frame/` and `recovered/` resources |
| `FRAME_DEPTH_WORKER` | Native worker executable |
| `FRAME_DEPTH_MODEL` | `primary` or `alternate`; the QNN list rejects alternate before opening hardware |

The QNN pinned deployment was measured around 23 ms warm. The older exact prepared reference engine was measured around 15.5 seconds. The newer approximately 9.68 ms primary benchmark recipe is a separate generated schedule/build identity; selecting the default backend does not install or select it. See [performance](../hexecutor/performance.md).

CLI selectors are `--depth-model`, `--depth-model-qnn`, and `--depth-model-prepared`. There is no hybrid backend or automatic Python graph fallback. The CPU reference is an explicitly separate offline tool.

## Geometry and state

The model images are uint8 320 × 256. The viewer defaults to spherical model inputs at 0.29 degrees per pixel; `--model-pinhole` selects the resized-pinhole mode and `--model-spherical` selects spherical input explicitly. Keep projection, eye order, and metric conversion coupled. The bridge's standalone `--spherical-deg` metadata does not replace C++ preprocessing.

Without host history, the bridge supplies zero hidden state and zero transforms. Host temporal mode supplies `[1,2,320,256]` float state and `[1,2,4,4]` transforms. `ModelTemporalState` rejects history after reset, missing alignment, an excessive gap, or a large rotation. The documented starting thresholds are 250 ms and 30 degrees; they are configuration choices, not established accuracy bounds.

The float output has two disparity planes. The byte head is treated as uncertainty, with a configurable rejection threshold; it is not a calibrated probability. The protocol can return the second disparity plane for state/recording. Keep output-head ordering explicit: the model's low-level output order and an outer wrapper's order are not interchangeable.

## Lifecycle

The C++ model adapter verifies pinned artifact hashes before acquiring the DSP and requires the preserving bootstrap/verified memory lease. The C++ side uses bounded socket waits, checks response shape and sequence, and stops publication on failure. `FRAME_DEPTH_TIMEOUT_S` defaults differ between the fast and reference engines.

An idle session closes after `FRAME_DEPTH_IDLE_CLOSE_S` (default 15 seconds) and reopens for the next request. This avoids the historical resident idle exit around 36.9 seconds that could leave the service stuck. Capture can close after each reply, but repeated reopen/close behavior has its own recorded failure history. Do not convert a cleanup failure into an invisible retry loop.

Source: [C++ pipeline](../../runtime/src/model_depth_pipeline.cpp), [native worker](../../runtime/tools/frame_depth_worker.cpp), and [temporal-state implementation](../../runtime/src/model_temporal.cpp). The separate temporal-fusion working documents are not maintained by this guide.

The live worker and executor host are C++17. Offline Python compatibility tools remain under `depth-model/python/depth_model/`; they are not a production fallback. See [native API](../hexecutor/api.md).
