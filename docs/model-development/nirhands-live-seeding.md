# NIR hand seeding check (2026-11-12)

The host viewer uses the `nirhands` checkpoint through `hand-tracking/tools/nirhands_replay.py` and the read-only XRService pair streamer from `hand-tracking/tools/stream_nir_pairs.cpp`. The capture implementation was copied from the `plaice` sibling worktree's `runtime/src/ir_pair_capture.cpp` and `runtime/src/v4l2_capture_clock.cpp`. It duplicates XRService descriptors with `pidfd_getfd`, reads completed V4L2 buffers with `VIDIOC_QUERYBUF`, and maps the DMA buffers read only. It does not attach, stop, or write to XRService. The live tool uses the generic rig file `stereo_nir_hands/config/steam_frame_tracking.yaml`; no per-device calibration was used for this run.

`HandSeeder` in `nirhands_live.py` accepts the frame pair and returns per-view pixel centres, side and score. `--seed mercury` is the default; `--seed mediapipe` retains the earlier palm plus landmark route. Mercury's detector and preprocessing come by import from `nirhands.frametop.teacher.MercuryTeacher.detect`: it points an upright virtual view along the Frame fisheye head-up direction, resamples to 160×160, and normalizes the grayscale mean to 0.5 and standard deviation to 0.25. The local ONNX files are from the existing `stereo_nir_hands/.deps/hand-tracking-models` directory. The `eval_frametop.py` Mercury baseline did **not** run this detector: it projected oracle label joints to set the keypoint crop. Its earlier score therefore says nothing about detector recall.

## CPU replay

From this repository's root, with the synthetic project kept read only:

```sh
NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
NIR_DATA=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/data
"$NIR/.venv/bin/python" hand-tracking/tools/nirhands_replay.py \
  --project "$NIR" --checkpoint "$NIR_DATA/runs/cotrain_hot3d_realbg_v4/last.pt" \
  --model-dir "$NIR/.deps/hand-tracking-models" --seed mercury --limit 50 --display
```

`--seed oracle --verify` is the crop/decode wiring check; it uses label centres and therefore is not an acquisition test. The live/replay seeder associates same-side detections from the two views by triangulation with a 12-pixel reprojection gate. A single-view detection uses a 0.45 m depth guess. Subsequent frames centre on the NIR model's joint 9; the detector is consulted again every ten frames or immediately after track loss. The 21-joint order matches the display's MediaPipe order.

## Detector benchmark

The command below evaluates 200 quality-1 labelled hands on 150 cached real Frame pairs, with a 100-pixel association gate at the labelled middle MCP. It runs CPU inference only and does not write camera images:

```sh
"$NIR/.venv/bin/python" hand-tracking/tools/benchmark_nir_seeds.py \
  --project "$NIR" --checkpoint "$NIR_DATA/runs/cotrain_hot3d_realbg_v4/last.pt" \
  --model-dir "$NIR/.deps/hand-tracking-models" --seed mercury --limit 200
```

| Measure | Result |
| --- | ---: |
| Left view matches / visible | 2 / 200 |
| Right view matches / visible | 7 / 200 |
| Stereo matches | 0 |
| Mono matches | 9 |
| Handedness after reversing detector side convention | 9 / 9 |
| Seed centre error, mean / median | 145.0 / 138.2 mm (9 hands) |
| NIR joint error with detector seed, mean / median | 132.3 / 124.7 mm (9 hands) |
| NIR joint error with oracle centre, mean / median | 14.9 / 13.2 mm (200 hands) |
| Detector latency per pair, mean / median on Mac CPU | 4.69 / 4.67 ms |

These are a detector failure, despite the fast inference. The side convention flip was established on only nine matches. The replay shows detections, including likely false positives; seeing a skeleton in its window does not establish correct tracking.

## Live launch when root-capable SSH is available

The Frame binary was compiled under `/tmp/nirhands-live` from the five C++ files in `hand-tracking/tools` and is the only device-side artifact. It has no installed service or persistent configuration. From the repository root on the Mac:

```sh
"$NIR/.venv/bin/python" hand-tracking/tools/nirhands_live.py \
  --project "$NIR" --checkpoint "$NIR_DATA/runs/cotrain_hot3d_realbg_v4/last.pt" \
  --model-dir "$NIR/.deps/hand-tracking-models" \
  --rig "$NIR/config/steam_frame_tracking.yaml" --seed mercury \
  --host root@frame --remote-binary /tmp/nirhands-live/stream_nir_pairs
```

The host argument must name an SSH account allowed to duplicate XRService's root-owned descriptors. With the owner's password, `--host frame --sudo` prompts locally and supplies it to transient remote `sudo -S`; it does not store the password or change device settings.

## Live capture check after the headset woke

On 2026-11-12 the streamer captured 50 live long-exposure pairs in `/tmp/nirhands-live/fifty.raw` on Frame. A compressed copy is in the ignored local `hand-tracking/.work/live-nir-20261112/fifty.raw.zst`; the uncompressed copy is beside it. Each pair contains two 1056×1024 monochrome images. The left sequence advanced from 449502 to 449600 and the right from 449504 to 449602. Capture period was mean 33.334 ms, median 33.332 ms, range 32.978–33.608 ms, or about 30 Hz. Left/right capture skew averaged 0.055 ms and peaked at 0.126 ms. Both images had nonzero brightness and changing pixels. XRService remained running after capture. Only `/tmp/nirhands-live/{stream_nir_pairs,*.cpp,*.hpp,one.raw,fifty.raw,fifty.raw.zst}` were created on the device; there was no installed service or device setting change.

`nirhands_live.py --host frame --sudo --max-pairs 2` ran end to end on the host, opened the two-view overlay window, and processed two live pairs with zero seeded hands in 23.5 and 10.7 ms respectively. A five-pair no-display run processed pairs in 4.5–17.7 ms, again with zero hands. The current SSH transfer carries 2.16 MB of raw pixels per pair and delivered only roughly 0.2–0.3 pairs per second in those bounded runs, with many source sequences skipped. Capture remains 30 Hz, but the raw SSH viewer is not real time. Capture-to-screen latency cannot be measured precisely because Frame's `CLOCK_MONOTONIC_RAW` and the Mac clock are not synchronized; the low delivered rate implies seconds of transport delay. A compressed or crop-level transport is needed for an interactive live test, while the purpose-trained detector is still pending.

## Compressed transport update

The default stream now sends full-resolution grayscale JPEGs at quality 70 in `FRNIR002` frames. Timestamps and sequences are retained; the receiver also accepts original raw dumps. `--raw` requests the lossless original protocol, and `--jpeg-quality 1..100` controls the bandwidth/fidelity tradeoff. A background receiver decodes continuously into a single pending slot, replacing older pairs while the display or inference is busy.

On ten cached labelled hands, JPEG70 averaged 152,945 bytes per pair versus 2,162,688 raw pixel bytes. Pixel mean absolute error averaged 2.41/255; running the NIR model at identical oracle centres changed joints by 0.90 mm on average, maximum 5.31 mm across 210 joints. Compression is lossy and this small check does not establish task accuracy for every scene.

A live 100-pair CPU run without a display delivered 13.12 processed pairs/s. A second 100-pair stream with the actual window displayed 47 newest pairs at 14.13 fps, skipping older received pairs during GUI startup/rendering. Typical receive-to-result time was about 5–9 ms, with occasional higher startup values. This is host decode-receipt to model-result timing, not capture-to-screen timing; the two clocks remain unsynchronized. Mercury still produced zero hands. XRService remained running and the capture helper exited after the bounded test.

Rebuild the temporary device binary when updating it:

```sh
scp hand-tracking/tools/{stream_nir_pairs.cpp,ir_pair_capture.cpp,ir_pair_capture.hpp,v4l2_capture_clock.cpp,v4l2_capture_clock.hpp} frame:/tmp/nirhands-live/
ssh frame 'cd /tmp/nirhands-live && g++ -O2 -std=c++17 stream_nir_pairs.cpp ir_pair_capture.cpp v4l2_capture_clock.cpp -lpthread -ljpeg -o stream_nir_pairs'
```

Use the live launch command above with `--host frame --sudo`; compression is enabled by default. The password is prompted locally and passed to transient remote sudo. No device settings or installed services are changed.

## End-to-end pipeline (2026-11-13)

`--seed nirhands` (now the default) replaces the Mercury seeder with the purpose-trained pieces from `stereo_nir_hands`: the per-view CenterNet hand detector, `seed_hands` stereo seeding, the stereo keypoint network, a tracker and One Euro smoothing. The code is [nirhands_pipeline.py](../../hand-tracking/tools/nirhands_pipeline.py) (pipeline and statistics), [nirhands_view.py](../../hand-tracking/tools/nirhands_view.py) (both views, a top-down head-frame depth panel and a HUD) and [nirhands_take.py](../../hand-tracking/tools/nirhands_take.py) (whole Frametop takes). The synthetic project stays read only. Default models are `data/runs/detector_s224/last.pt` (all contributors; its log stops at epoch 33 of 40) and `data/runs/cotrain_v5_labels_fah/best.pt` (L2).

Per pair: the detector runs every frame while any hand is missing or tentative, otherwise every third frame. Stereo seeds associate with tracks within 10 cm in 3D. Single-view seeds associate in that image, within 0.8 detected radii of the projected track centre: their depth comes from the detector's distance head and is often 5–10 cm off, which in 3D made a hand's own seed miss its track every frame. A new track needs a stereo seed scoring at least 0.35 or a single-view seed scoring at least 0.5. It is shown once it has two detector hits, mean joint visibility of at least 0.45, and a median index-to-pinky knuckle span of at least 4.5 cm over at least three frames. At most two hands are shown, and up to two more tentative tracks may exist, so a rejected false positive cannot block a real hand. Each track is cropped at its last middle MCP plus a constant-velocity step, and all tracks run as one keypoint batch. A track is dropped after three detector runs and 0.3 s without a matching seed (1 s while mean visibility is at least 0.6), after three weak frames (low visibility, or the root landing more than 12 cm from the crop centre), or when its knuckle span over the last ten frames has a median under 3.5 cm. Two tracks are one hand when both their middle MCPs and their wrists are within 6 cm: a mirrored left/right reading of one hand shares both, while touching hands do not. A hand re-found within 1 s and 15 cm of a dropped same-side track keeps its ID.

Where the thresholds came from:
- **Seed scores:** on the Frametop no-hands take, stereo seeds scored at most 0.34 (90th percentile) and single-view seeds at most 0.41.
- **Knuckle span:** on real Frametop hands the per-frame 5th percentile is 4.6–6.4 cm. A collapsed skeleton on a bright window in the first live session had a median of 3.0 cm.

### Live

The streamer binary was rebuilt on 2026-11-13 under `/tmp/nirhands-live` (lost on reboot; rebuild as above). From the repository root:

```sh
NIR=/Users/owenelliott/orca/workspaces/synthetic-hand-render/cowfish/stereo_nir_hands
"$NIR/.venv/bin/python" hand-tracking/tools/nirhands_live.py --project "$NIR" \
  --rig "$NIR/config/steam_frame_tracking.yaml" --host frame --sudo \
  --record hand-tracking/.work/live-nir-$(date +%Y%m%d-%H%M).frnir
```

Keys: `q` quit, space pause (`n` steps while paused), `d` detector circles, `r` raw instead of smoothed joints. `--record` saves the received stream; `--input FILE [--realtime]` replays it (or `.raw`/`.raw.zst` dumps) through the same receiver, in order. `--session-calibration DIR` uses a Frametop-format session's factory calibration instead of the generic rig. Stereo pairing gates rays at 4 cm, so a poor generic rig shows up as single-view (`mono`) hands and missed acquisitions. `--json FILE` writes the statistics summary on exit.

### Offline checks

The cached 50-pair live capture (`fifty.raw.zst`, a desk with no hands in view) gives zero tracks at 16.6 ms per pair. Before the start thresholds, a 0.42 detection on the laptop could seed a track; it no longer does. Packing 200 frames of a Frametop take into the JPEG70 `FRNIR002` stream and replaying it through `nirhands_live.py --input --record` tracked hands in 198 of 200 frames, and the recording was byte-identical to its input.

Whole takes of held-out contributor `884acd76` (first 600 frame sets each), with `detector_lopo_s224` (trained without this person). The keypoint model is L2 as above. M4 Pro, MPS:

| Take | Frames | With hands | Track IDs | Drops | Jitter raw / smoothed (mm RMS) | Median ms |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 02-pose-sweeps | 600 | 578 | 5 | 3 missed | 3.7 / 1.9 | 25.6 |
| 03-gestures | 587 | 445 | 7 | 8 missed, 2 duplicate | 8.1 / 5.1 | 27.0 |
| 04-desk-work | 282 | 279 | 3 | 1 missed | 7.5 / 4.1 | 25.0 |
| 05-objects | 302 | 275 | 8 | 8 missed | 21.6 / 16.3 | 24.8 |
| 08-no-hands | 131 | 0 | 0 | — | — | 15.4 |

Jitter is the RMS second difference of joints between consecutive frames, so real acceleration counts too; compare configurations on the same footage, not as an absolute. "With hands" counts frames with at least one confirmed hand, not recall against labels. Track IDs over two hands measure fragmentation. These are tracking-behaviour checks; they do not score joint accuracy. Scoring whole takes against Mercury run the same way (handoff step 2) is still to do.

```sh
"$NIR/.venv/bin/python" hand-tracking/tools/nirhands_take.py --project "$NIR" \
  --contributor 884acd76 --take 03-gestures --display      # or --video out.mp4 --json out.json
```

Frametop images are CC BY-NC 4.0 with privacy terms; keep rendered videos in ignored local directories such as `hand-tracking/.work/`.

### First live session (2026-11-13 14:14)

The recording is `hand-tracking/.work/live-nir-20261113-1414.frnir`, made with the generic rig. A 14:13 attempt recorded nothing. Replay with `--input`: the final pair is cut off when the stream stops, and replay now ends cleanly there.

- **Transport:** 334 pairs in 66 s, about 5 pairs/s (median gap 100 ms). One gap of 718 sequence numbers is a stall of about 12 s. The recording holds every pair received, so this is the SSH/streamer path, not Mac inference (about 25 ms per pair).
- **Tracking:** with the rules above, confirmed hands in 267 of 334 pairs, 8 track IDs. Both hands in front of the keyboard get skeletons on them in both views.
- **False positives found and fixed:** a bright window patch produced two confirmed false tracks. One was a stereo seed at 0.55 with visibility 0.25–0.31; the other a single-view seed at about 0.5 with a 3.0 cm knuckle span. The visibility and knuckle-span gates remove both.
- **Remaining:** brief side flips and occasional same-side pairs on single-view hands.
- **Detector domain gap:** at threshold 0.3, the detector fired 496 times on the left view (median score 0.41) and 116 times on the right (0.34). On Frametop takes the two views are balanced: 140/140 at 0.67/0.64 for desk-work, and 187/175 at 0.57/0.53 for gestures. JPEG70 changes neither count nor score. Rotating the live right view by 180° lowers its count to 24, so orientation is right. The weak right view makes most live hands single-view, which drives the remaining side flips and fragmentation. The likely cause is the new person, room and headset rather than the transport. Pseudo-labelling recorded live pairs with the FAH/WiLoR labeller and fine-tuning the detector on them is the direct test.

## Crop refinement and split smoothing (2026-11-13)

Two tracker changes, both on by default; `--legacy` restores the earlier behaviour, which the native C++ port still implements (`compare_native_nirhands.py` uses `TrackerConfig.legacy()`).

- **Crop refinement** (`--refine-m 0.015`): when the predicted middle MCP lands more than 1.5 cm from the crop centre (and within the 12 cm jump gate), the hand is cropped again at that point and re-run. Training only jittered the crop centre by 1 cm laterally and 3 cm in depth, so a constant-velocity crop at slow or uneven frame rates puts the hand off-centre by more than the network ever saw.
- **Split smoothing** (`--smoothing split`): One Euro filters on the palm pose (wrist position, palm axes) and separately on hand-local joints (`--finger-min-cutoff 2.5 --finger-beta 20`). A hand moving rigidly with a fixed pose keeps that pose exactly; world-joint smoothing lags fingertips more than the wrist when the hand rotates.

`nirhands_take.py --labels <data>/frametop_labels/fah_v1` now scores tracked hands on the labelled frame sets (`nirhands_metrics.py`): recall, MPJPE, PA-MPJPE and per-finger flexion (sum of bend angles; error, correlation and spread ratio against the label). "Local jitter" in the statistics is jitter in the palm frame, so whole-hand motion drops out.

Test contributors 08466c63, 1ae30f06 and c888a731, takes 02/03/04, FAH labels at quality 1 (8–47 hands per take, so differences under about 1 mm are noise):

- **Refinement** cut raw hand-local jitter on most takes (08466c63 gestures 24.9 → 20.1 mm, 1ae30f06 desk-work 9.4 → 7.4 mm) and track fragmentation (08466c63 pose-sweeps 83 → 45 track IDs, 1ae30f06 pose-sweeps 11 → 8), with recall up 2–3 points. PA-MPJPE moved by under 0.6 mm either way. It costs about 10 ms per frame when it fires.
- **Split smoothing** was neutral on these metrics: smoothed local jitter within ±0.6 mm of world smoothing, accuracy unchanged. Its benefit (pose held during rigid motion) is not measured by these takes.
- **Per-finger flexion** barely depends on the tracker: correlation with labels about 0.8 for the four fingers and 0.5 for the thumb, mean absolute error 19–29°, in every configuration. The finger failure is in the keypoint model (and possibly its FAH labels), not the tracking.
- **c888a731** has 30–60 mm jitter and 25–40 track IDs per take in every configuration: a person-specific model failure, like the live session's.

## Skeleton checkpoints (2026-11-13)

`nirhands_live.py`, `nirhands_take.py` and `nirhands_replay.py` now accept a `SkeletonHandNet` checkpoint (`args["arch"] == "skeleton"`, see `stereo_nir_hands/docs/handoff_2026-11-13_skeleton.md`) through the same `--checkpoint` flag. Nothing else changes on the command line; the startup line prints `(skeleton model)` or `(keypoint model)`.

- **How:** `load_keypoint_model` and `run_keypoint_model` in [nirhands_pipeline.py](../../hand-tracking/tools/nirhands_pipeline.py) load either model and, for a skeleton model, pass each crop's geometry. Its `keypoints` and `disparity` outputs are then the skeleton joints (mean skeleton times predicted scale), which decode like the keypoint model's. Without the geometry the model returns its auxiliary heatmap path, which is not the deployable output.
- **Check:** `nirhands_replay.py --seed oracle --verify` agrees with `real_eval.predict_model` to 0.0000 mm for both model types, on CPU and MPS.
- **Fetch a checkpoint from the training PC:**
  ```
  mkdir -p "$NIR/../data/runs/skeleton_partial_v1"
  scp "owens@100.111.38.45:C:/Users/Owens/cowfish/data/runs/skeleton_partial_v1/best.pt" "$NIR/../data/runs/skeleton_partial_v1/"
  ```
  then add `--checkpoint "$NIR/../data/runs/skeleton_partial_v1/best.pt"` to any command above.
- **Tracker checks that behave differently:** the skeleton's palm is a rigid template times a scale of 0.75 to 1.25, so the index-to-pinky knuckle span cannot collapse. `confirm_span_m` and `keep_span_m` therefore never reject a skeleton-model track; false positives are left to the visibility and detector checks.
- **Not done:** `export_nirhands_onnx.py` and the native C++ port still assume `StereoKeypointNet`.

Take `884acd76/03-gestures` (all frame sets, `detector_s224`, labels `fah_v1`, MPS), with the skeleton run's `last.pt` after 2 of 8 epochs. This is a plumbing check, not a result:

| | L2 `best.pt` | skeleton, epoch 2 of 8 |
|---|---|---|
| MPJPE, median, smoothed | 8.9 mm | 16.0 mm |
| PA-MPJPE, median, smoothed | 6.0 mm | 8.5 mm |
| Keypoint step, median (first 400 sets) | 20 ms | 27 ms |
| Hand-local jitter, raw / smoothed (first 400 sets) | 5.8 / 3.6 mm | 3.6 / 2.2 mm |

**Native replay (2026-11-13, 22:30).** `nirhands_replay` and the native viewer load a skeleton export (`export_nirhands_onnx.py` with a skeleton checkpoint) on the CPU path. Against `nirhands_pipeline.py` on 150 pairs of `live-1414.raw`, raw joints agree to p50 1.4 mm but reach 153 mm on tracks about to be dropped, and track IDs differ on 23 frames; the cause is not established. See [the integration handoff](nirhands-skeleton-integration-handoff-2026-11-13.md).

## Per-user hand profile (2026-11-14)

A skeleton checkpoint poses one mean hand at a predicted scale. Its finger bones are about 5% longer relative to the palm than the MANO device labels, so on a new person the fingertips land past the real tips. `--profile FILE` (live viewer and take replay, skeleton checkpoints only) gives the model the wearer's own palm shape and bone lengths through its `profile` input instead.

[nirhands_profile.py](../../hand-tracking/tools/nirhands_profile.py) measures the profile from a recorded stream: it runs FAH-L on both views of sampled frames, triangulates, and takes the median over hands whose views agree within 1.5 px. It needs the FAH venv. Record 20–30 s of open, spread hands first.

```sh
"$NIR/.deps/fah_venv/bin/python" hand-tracking/tools/nirhands_profile.py --project "$NIR" \
  --rig "$NIR/config/steam_frame_tracking.yaml" --checkpoint "$NIR_DATA/runs/skeleton_tight96_v1/best.pt" \
  --input hand-tracking/.work/live-nir-<stamp>.frnir --output hand-tracking/.work/hand-profile.json
```

Checked on the live recording `live-nir-20261114-1102-tight96.frnir` with a profile measured from the separate 11:01 recording (61 right hands). The reference is FAH-L triangulated on 348 hands, so numbers are against that labeller, not ground truth:

| `skeleton_tight96_v1` | Mean hand | Wearer's profile |
| --- | ---: | ---: |
| MPJPE / after rigid alignment | 16.5 / 10.0 mm | 14.6 / 8.7 mm |
| Fingertip image error | 17.6 px | 13.8 px |
| Tips more than 5 mm past, by finger | 40–75% | 5–25% |
| Finger bend error, index to pinky | 20.5–23.2° | 19.2–23.7° |

The loose 128 px model gains the same way (15.2 → 13.6 mm). Finger bend does not improve: curled fingers are still read about 20° too open. The profile was measured on right hands only and applied to both.
