# Datasets, replay, and evaluation

The dataset harness records raw observations and the provenance needed to replay them. Its unit is a session with a stable scene ID and a development, tuning, or held-out split. The index rejects assigning one scene to conflicting splits; adjacent frames from one room should not quietly cross that boundary.

## Capture

`scripts/room_dataset.py capture` coordinates camera/pose recorders and optional controller markers and IR. The Steam launcher (`launch_room_dataset.sh`) uses the live QNN depth viewer so RGB and depth share one acquisition path. Running a second color-camera recorder beside that viewer has caused missed pairs.

The launcher records until stopped unless a cap is configured. Holding trigger and grip together requests a clean stop. `FRAME_DATASET_FRAMES`, `FRAME_DATASET_SPACING_MS`, and `FRAME_DATASET_MIN_FREE_GB` configure limits. Raw data can consume roughly 228 MB/s in the recorded RGB-plus-IR configuration; the free-space stop is operationally significant.

Controller marks store poses, timestamps, a class, and strokes. A controller's device origin is not a measured surface-contact point. Supply a measured `--controller-tip` offset before using those marks as a metric reference.

| Session entry | Purpose |
| --- | --- |
| `rgb/` | Raw NV12, rectified/model inputs, timing and calibration |
| `ir/` | Paired tracking-camera frames and optional refined depth |
| `head_pose.csv`, `marks.csv` | Poses and controller annotations |
| `model/` | Both disparity/uncertainty planes and batch provenance |
| `depth/` | Replayable depth, validity, confidence, and camera pose |
| `mesh/`, `tags/` | Reconstructed surfaces and optional target detections |
| `session.json`, `manifest.json` | Configuration, completion state, and integrity hashes |
| Logs and pulse/thermal records | Acquisition, lifecycle, and environmental context |

## Finalize, transfer, and replay

From `runtime/` in the appropriate host/headset environment:

```sh
python3 scripts/room_dataset.py finalize SESSION
python3 scripts/room_dataset.py validate SESSION
python3 scripts/room_dataset.py index DATASET_ROOT
python3 scripts/pull_room_dataset.py --session SESSION --raw-only
python3 tools/mesh_viewer.py --root DATASET_ROOT --open
```

Use the configured model Python environment for inference. Offline batch inference now defaults to `prepared-qnn`; it no longer invokes the removed hybrid path. `--skip-inference` reuses supplied outputs and preserves their recorded backend provenance when available. Validation checks sizes/replay compatibility and refreshes the manifest; it is not an accuracy certificate.

The pull tool resumes copies and checks copied files against hashes. Large raw camera files stay on the headset unless explicitly requested. Older FFV1-packed sessions can be decoded by `room_video.py`; current capture retains individual raw files. Partial sessions remain partial rather than being filled with synthetic observations.

## IR emitter and refinement

BELICE pulses, paired IR frames, and sham cycles provide a separate diagnostic depth path. The timing sweep measures a useful pulse lead; an unmeasured fallback lead must remain marked as such. Current refinement uses pulse coverage, contrast in both eyes, and comparison with sham statistics. Missing emitters, inadequate sham coverage, or old pulse metadata should prevent refinement, not silently produce trusted IR depth.

IR-to-RGB projection needs the session's measured stereo yaw, extrinsics, and exposure pose. Dark-neighbor subtraction and prior-guided refinement affect the estimator. A combined RGB+IR mesh is a diagnostic comparison until its geometry and timing have been independently accepted.

Compositor loss and DSP cleanup failures are retained in session logs. The launcher blocks another NPU capture in a boot marked by failed cleanup. After reboot, restore ptrace/memory prerequisites before recording again.

Source: [harness](../../runtime/scripts/room_dataset.py), [offline inference](../../runtime/scripts/model_offline_batch.py), [transfer](../../runtime/scripts/pull_room_dataset.py), and [IR refinement](../../runtime/scripts/ir_dot_lib.py).
