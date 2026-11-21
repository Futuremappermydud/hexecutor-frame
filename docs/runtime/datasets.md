# Datasets, replay, and evaluation

The dataset harness records raw observations and the provenance needed to replay them. Its unit is a session with a stable scene ID and a development, tuning, or held-out split. The index rejects assigning one scene to conflicting splits; adjacent frames from one room should not quietly cross that boundary.

## Capture

`scripts/room_dataset.py capture` coordinates camera/pose recorders and optional controller markers and IR. The Steam launcher (`launch_room_dataset.sh`) uses the live QNN depth viewer so RGB and depth share one acquisition path. Running a second color-camera recorder beside that viewer has caused missed pairs.

Build the recorder on Frame from `depth-runtime/` with `make room-dataset OPENXR_INCLUDE=/path/to/openxr-sdk/include`. Run `python3 scripts/install_room_dataset_menu.py` from that deployment to add the Steam shortcut. After a reboot, run `scripts/setup_frame.sh --qnn` from the Mac to prepare camera access and verify the DSP memory lease. Set `FRAME_DEPTH_PACKAGE` when the validated package is installed outside its default location. The capture coordinator uses a Python environment with NumPy, which can be selected with `FRAME_DATASET_PYTHON`. Live inference stays in the native `frame_depth_worker`.

The launcher records until stopped unless a cap is configured. Holding trigger and grip together requests a clean stop. `FRAME_DATASET_FRAMES`, `FRAME_DATASET_SPACING_MS`, and `FRAME_DATASET_MIN_FREE_GB` configure limits. Raw data can consume roughly 228 MB/s in the recorded RGB-plus-IR configuration; the free-space stop is operationally significant.

For either capture mode, `room_dataset.py capture --no-color` skips the full-resolution NV12 pair while retaining the rectified, spherical, and pinhole luma used by offline inference and tag detection. `--keep-floods` keeps the current IR flood mode throughout capture. Otherwise, the recorder restores the mode it read at startup.

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

From `depth-runtime/` in the appropriate host/headset environment:

```sh
python3 scripts/room_dataset.py finalize SESSION
python3 scripts/room_dataset.py validate SESSION
python3 scripts/room_dataset.py index DATASET_ROOT
python3 scripts/pull_room_dataset.py --session SESSION --raw-only
python3 tools/mesh_viewer.py --root DATASET_ROOT --open
```

Use the configured model Python environment for inference. Offline batch inference now defaults to `prepared-qnn`; it no longer invokes the removed hybrid path. `--skip-inference` reuses supplied outputs and preserves their recorded backend provenance when available. Validation checks sizes/replay compatibility and refreshes the manifest; it is not an accuracy certificate.

The pull tool defaults to the recorder's `depth-runtime/experiments/room-dataset` directory on Frame. Use `--remote-dir`, `--remote-root`, and `--remote-python` for a custom deployment. The pull tool resumes copies and checks copied files against available manifest hashes. Large raw camera files stay on the headset unless explicitly requested. Older FFV1-packed sessions can be decoded by `room_video.py`; current capture retains individual raw files. Partial sessions remain partial rather than being filled with synthetic observations.

## IR emitter and refinement

BELICE pulses, paired IR frames, and sham cycles provide a separate diagnostic depth path. The timing sweep measures a useful pulse lead; an unmeasured fallback lead must remain marked as such. Current refinement uses pulse coverage, contrast in both eyes, and comparison with sham statistics. Missing emitters, inadequate sham coverage, or old pulse metadata should prevent refinement, not silently produce trusted IR depth.

IR-to-RGB projection needs the session's measured stereo yaw, extrinsics, and exposure pose. Dark-neighbor subtraction and prior-guided refinement affect the estimator. A combined RGB+IR mesh is a diagnostic comparison until its geometry and timing have been independently accepted.

Compositor loss and DSP cleanup failures are retained in session logs. The launcher blocks another NPU capture in a boot marked by failed cleanup. After reboot, restore ptrace/memory prerequisites before recording again.

The dataset index, offline depth pass, transfer helper and IR refinement live under `depth-runtime/scripts/`. Keep capture metadata, model artifact identities and replay parameters with the dataset.

For recordings collected across rooms or devices, retain rig identity, per-eye timing, and exposure-aligned poses so calibration and depth errors can be evaluated independently. Keep validation sessions disjoint from tuning sessions and review recordings for identifying content before sharing.

## Offline IR yaw normalization

`depth-runtime/tools/ir_yaw_normalization.py` scores saved motion-stereo matches using each right-image ray's spherical yaw sensitivity. It leaves feature matches and their support unchanged. The right rectifier's fixed roll is included; an image-wide pixels-per-degree scale is only correct for a restricted set of rays. Python here is offline analysis and is not part of the live depth path.

From `depth-runtime/`, using a Python environment with NumPy:

```sh
python3 tools/ir_yaw_normalization.py --samples /path/to/motion-stereo-samples.npy --out /path/to/yaw-results --correction-deg 0
python3 -m unittest discover -s tools -p test_ir_yaw_normalization.py
```

The input format is documented in the tool's module docstring. Supply the recorded image dimensions, angular step and right-camera roll if they differ from the defaults. `--correction-deg` defaults to zero and can apply an explicitly supplied, previously validated additive correction; the tool never fits it. Output records input/source hashes, frame-resampled uncertainty, distance-bin coverage and spread. Missing distance bins remain missing. The scorer does not declare acceptance: compare against an independent reference and a frozen protocol, and keep development results separate from fresh held-out evidence.
