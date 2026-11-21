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

For RGB timestamps, the capture-clock helper first checks XRService for a descriptor matching the requested 2464 × 2464 NV12 node. It can duplicate the matching descriptor from another owner when a separate camera service streams the pair.

Dataset processing associates a camera pose with the estimated exposure and rejects missing or excessively distant poses. The historical 8 ms exposure offset is configurable and is not a universal calibration. Compositor restart, camera sequence gaps, and uncertain clock alignment must remain visible in metadata.

Source: [capture](../../depth-runtime/src/dma_capture.cpp), [rectification](../../depth-runtime/src/depth_pipeline.cpp), [calibration definitions](../../depth-runtime/include/calibration.hpp).

The capture-clock helper maps V4L2 and raw monotonic timestamps into the runtime clock domain. Verify its calibration against measured exposure timing for the camera stream under test. [Capture-clock code](../../depth-runtime/src/v4l2_capture_clock.cpp)

## Offline stereo self-calibration

The tools under `depth-runtime/tools/` can assess stereo alignment from natural features and recorded headset motion. Keep the nominal rectifier and the fitted lens model distinct, and score horizontal prediction on independent recordings in a common camera frame. Small fitted residuals can coexist with a biased yaw estimate. Use an independent reference and report its uncertainty alongside the candidate error. The experimental estimates have not met the criteria for replacing the live calibration. [Motion-stereo replay](../../depth-runtime/tools/replay_ir_motion_yaw.cpp)

## IR yaw scan viewer

`frame_yaw_viewer` is a passthrough prototype for the IR tracking-camera pair. Build and run it on Frame from `depth-runtime/` with SteamVR running, the normal tracking-camera capture permissions, and the external IR dot emitter off.

```sh
make yaw-viewer OPENXR_INCLUDE=/path/to/openxr-sdk/include
bin/frame_yaw_viewer --output captures/yaw-room-01 --seconds 60
```

Recording starts when the viewer becomes visible. Move slowly sideways while keeping textured objects both nearer and farther than 2 m in view. **Fit found / still recording** marks the first usable progress fit, which uses 32 keyframes. At the time limit, the viewer fits the full recording with every pair as a keyframe candidate (about 30 s for a minute of pairs). **Yaw estimate ready** means the final fit has enough tracked matches, the two time halves agree within 0.087°, and the matches nearer and farther than 2 m agree within 0.087°. **Need near and far surfaces** means one side of 2 m has fewer than 100 matches or 8 keyframes. Send `SIGUSR1` to end recording early and fit the captured pairs, or exit the viewer to save the captured pairs and cancel any fit still running. Relaunch for a new scan.

The capture contains up to five IR pairs per second, their timestamps, the rectified camera poses and spline calibration in `problem.txt`, and the outcome in `result.json`. Allow about 650 MB for a full minute. `samples.csv` records the latest completed fit's matches. `problem.txt` replays the final fit. When the saved fit ran on a shorter prefix or fewer keyframes, `fit-problem.txt` describes it. Replay either problem file with `bin/replay_ir_motion_yaw PROBLEM.txt OUTPUT.csv` to regenerate the matches and the original median-based estimate.

The yaw fit holds the tracked translation scale fixed, gives each keyframe equal initial weight before robust weighting, and fits a right-camera roll correction together with yaw. The rectification assumes fixed corrections of 0.4° roll and 0.25° pitch on top of the factory calibration. A roll error shifts horizontal disparity in proportion to image height, and because far matches usually sit high in the image it otherwise reads as a depth trend. On nine replayed scans from 2026-11-20 the fitted roll correction was −0.19° to −0.35°. The roll from the vertical offset of the stereo matches alone (`stereo_rows`, which needs no poses) agreed, at −0.18° to −0.26°, and the near/far differences fell below 0.05°. The roll term did not reduce the spread of yaw between scans. The same row offsets also give a yaw estimate that needs no poses or motion depth: a yaw rotation shifts rows in proportion to the product of the horizontal and vertical image angles. On 16 IR scans from 2026-11-05, -06 and -20, motion-stereo yaw minus row yaw averaged −0.156° (sd 0.032° per scan, r = 0.89) while the yaw itself ranged over 0.3°. `stereo_rows` reports the row yaw and the row yaw plus that offset (`calibrated_yaw_correction_degrees`), and the HUD shows the calibrated value. The offset probably comes from lens-model error at the image corners and should be re-checked after any IR recalibration. Because both estimates follow the same changes between scans, those changes are movement of the stereo rig rather than estimator noise. `result.json` also reports yaw without the roll term, yaw for the near and far sides of 2 m and for the 0–1.2, 1.2–2, 2–3.5 and beyond-3.5 m bands (with roll held at the fitted value), and a yaw, roll and scale fit when the matches span enough depths. A bias in tracking scale or exposure timing can still shift the reported yaw. The time and depth checks measure consistency within the recording, not accuracy.

SteamVR keeps only about 0.2 s of pose history; a lookup for an older time returns the oldest stored pose. The viewer skips a pair when its first pose lookup comes more than 100 ms after the pose time, and records each lookup delay in `pairs.csv` and `result.json` (`pose_lookup`). Each kept pose is looked up again about 60 ms later. `poses-requery.csv` lists the differences, `result.json` summarizes them under `pose_requery` (lookups more than 150 ms after the pose time count as outside the history window), and `problem-requery-after_60ms.txt` replays the fit with the second lookups when every one was located inside the window. A difference shows that the pose for a past time changes after it is first returned.

The viewer samples every Linux thermal zone every 5 s into `thermal.csv`, continuing through the final fit. `result.json` (`thermal`) gives start, end, minimum and maximum over the recording for the camera, always-on, CPU, DDR and battery zones. All of these are chip-side sensors that warm within seconds; the camera mount and optics lag them. The battery zone reads the external battery pack, not the headset body. The IR stereo yaw has drifted by about 0.35° in 15 minutes before, so compare yaw across scans against these temperatures and the time since power-on before treating the difference as estimator noise.

At capture the viewer also looks up each pose at exposure offsets from 0 to 20 ms in 2 ms steps and writes `problem-offset-<ms>ms.txt` for every offset whose poses were all located. Score them offline on Frame with `bin/replay_ir_motion_yaw problem-offset-8ms.txt samples-8ms.csv` followed by `bin/fit_yaw_scan problem-offset-8ms.txt samples-8ms.csv`. A timing error lowers the number of accepted matches and the mean sweep score, and raises `residual_scale`.

On 2026-11-20, replaying all recorded pairs from eight successive scans in one session gave a 0.117° yaw range with fixed scale and balanced keyframes, against 0.413° with a separate scale fit per scan. The largest difference between successive scans was 0.049° against 0.333°. Both fits used the same replay matches. This comparison measured repeatability on development recordings. An independent geometric reference is still needed to measure yaw accuracy.

The viewer reads `/persist/xrservice.json` and `/persist/device_config.json` by default. `--xrservice` and `--device-config` select explicit snapshots. It locates tracked OpenXR poses at the left camera timestamp minus `--exposure-offset-ms`, which defaults to 12 ms. In the 0–20 ms pose-offset sweep of one 2026-11-20 scan, match count and residual were best near 12.5 ms; yaw moved by only about 0.002° per ms. The reported value is a signed IR yaw correction in degrees. It is diagnostic and is never written into system calibration. Capture and fitting use native C++ workers while OpenXR renders the HUD. [Viewer entry point](../../depth-runtime/tools/frame_yaw_viewer.cpp)

### Status and open tests (2026-11-20)

Development evidence on one headset:

- **Scan-to-scan changes are rig movement.** Yaw, roll and pitch from row offsets need no poses or motion depth, yet they change between scans by more than their noise. Row yaw follows motion-stereo yaw (r = 0.89 over 16 scans on three days).
- **Ruled out as the source of the remaining scatter:** lever arm, camera-to-head rotation, baseline direction, exposure offset, pose re-query, motion blur, sub-pixel locking and surface bias. None reduced it.
- **Cold start.** After the headset sat unused, motion-stereo yaw moved from +0.085° to −0.071° over 8 minutes in the same room. Row yaw moved less, and the gap between the two estimates shrank from 0.19° to about 0.07°. Wait about 15 minutes after a cold start before a scan meant for use.
- **RGB pair.** The 2026-11-05/06 room-dataset sessions show RGB row offsets that also change between sessions and correlate with IR yaw (6 sessions).

Not yet tested:

- the complete warm-up curve, and whether the row-yaw offset depends on temperature;
- a cold-start calibration that combines continuous sliding-window estimates with a warm-up model (not built);
- the effect of re-seating the headset or changing strap tension;
- row yaw with the head still, which needs a standalone stereo matcher;
- absolute accuracy against a metric reference (tracking scale and yaw);
- RGB yaw from motion stereo, which needs the rolling-shutter readout time;
- removing the fixed 0.4° roll and 0.25° pitch rectification constants.
