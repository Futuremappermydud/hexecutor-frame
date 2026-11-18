# Spatial runtime

The runtime acquires a calibrated stereo pair, computes depth off the render thread, and publishes geometry with a capture-time pose. It supports a diagnostic OpenVR app, OpenXR point-cloud/mesh presentation, recording/replay tools, and an opt-in occlusion API layer.

| Source | Responsibility |
| --- | --- |
| `src/dma_capture.cpp`, `vk_detile.cpp` | Borrow camera buffers and detile the image planes |
| `src/depth_pipeline.cpp` | Rectification, calibration selection, CPU depth path |
| `src/vulkan_stereo_pipeline.cpp` | Native GPU disparity and validity filtering |
| `src/model_depth_pipeline.cpp` | Learned-depth process, protocol, temporal inputs, and metric output |
| `src/cloud_worker.cpp` | Background capture, backend selection, observations, and fusion |
| `src/tsdf_volume.cpp`, `persistent_depth.cpp` | Persistent geometric representations |
| `src/openxr_point_cloud.cpp`, `overlay_renderer.cpp` | Headset presentation |
| `src/openxr_occlusion_layer.cpp`, `occlusion_renderer.cpp` | API-layer interception and occlusion rendering |
| `scripts/room_dataset.py` and replay tools | Repeatable data collection and evaluation |

All paths in this table are relative to `runtime/`. Read [cameras](cameras.md) before changing geometry, [model integration](model-depth.md) before changing the bridge, and [rendering](rendering.md) before changing coordinate or submission behavior. The [system overview](../architecture.md) explains the component boundaries.

See [the OpenXR hand joint viewer](hand-tracking-viewer.md) for a simple in-headset camera/joint view, build instructions and session lifecycle.
