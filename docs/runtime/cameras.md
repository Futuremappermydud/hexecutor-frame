# Cameras, calibration, and timing

## Sensor identity and buffer ownership

The color stereo pair is Sony IMX616: 2464 × 2464 NV12 UBWC, with 9,564,160-byte DMA slots in the measured configuration. OG01A1B cameras are separate GREY tracking/IR sensors. `/dev/video99` is a compositor spectator view and cannot substitute for a stereo pair.

`DMACapture` discovers the live compositor PID and its DMA-BUF descriptors, duplicates descriptors through `pidfd_open`/`pidfd_getfd`, and maps the shared buffers. Descriptor numbers are not stable across compositor restarts. The application must retain its own references and rediscover after a restart; it must not hardcode the example descriptor numbers from old logs.

The sensor devices are owned by XRService/the compositor. Opening claimed `/dev/video*` sensors directly is not the acquisition path. Vulkan detiling converts UBWC planes into the image representation consumed by rectification. Luma supplies stereo matching; UV supplies color where requested. Avoid copying whole raw frames across processes: the learned bridge receives compact model inputs, not camera DMA buffers.

## Rectification and coordinate frames

Factory camera intrinsics use a Kannala–Brandt fisheye model. The right color camera is mounted 180 degrees inverted. Its rectification ray uses `(-rx, -ry, 1)` where the left uses `(rx, ry, 1)`. Omitting that optical parity correction destroys epipolar alignment.

The shared rectified image is 616 × 616. Metric pinhole depth obeys `Z = focal × baseline / disparity` with the focal length and baseline of the rectified rig. Model inputs use their own resized or spherical projection; use the corresponding conversion rather than applying the pinhole formula blindly.

| Frame | Convention |
| --- | --- |
| Rectified camera | x right, y down, z forward |
| OpenVR/OpenXR world presentation | x right, y up, forward along negative z in the documented standing/stage convention |
| `TsdfPose` | Explicit camera-to-world rotation and translation in metres |

Compose camera-to-head calibration with the headset pose before integrating a sample. Do not treat a camera-frame point as already in stage space.

## Calibration selection

`DepthPipeline` records the calibration source, selected history entry, and policy. Stereo history defaults to the medoid: the entry with the smallest summed rotational distance to the others. `FRAME_STEREO_CALIBRATION_POLICY=latest` restores newest-entry selection. The front-camera-to-head alignment follows its separately handled live history.

The repeated finding was that the newest stereo estimate is noisy, while XRService applies stereo estimates only after its own agreement checks. Taking the newest entry unconditionally made disparity sensitive to that noise. On the recorded history, medoid selection reduced disagreement with the history consensus from about 0.47 to 0.08 pixels. That measures agreement with a consensus, not absolute depth accuracy.

Tag observations also showed short-timescale yaw drift outside the history spread. Medoid selection does not solve that drift. Calibration parameters, their capture time, and the chosen entry must accompany a dataset; a new depth model cannot compensate for an unrecorded change in rig geometry.

## Time alignment

`timed_stereo_capture`, `v4l2_capture_clock`, and the IR capture tools retain sequence/timestamp information. Prefer measured exposure-related timing over whichever headset pose is newest at processing completion. A DMA read midpoint is a fallback estimate, not proof of exposure time.

Dataset processing associates a camera pose with the estimated exposure and rejects missing or excessively distant poses. The historical 8 ms exposure offset is configurable and is not a universal calibration. Compositor restart, camera sequence gaps, and uncertain clock alignment must remain visible in metadata.

Source: [capture](../../runtime/src/dma_capture.cpp), [rectification](../../runtime/src/depth_pipeline.cpp), [calibration definitions](../../runtime/include/calibration.hpp).

Capture-clock development in `runtime/src/v4l2_capture_clock.cpp` remains local work in progress.
