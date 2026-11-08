# Room reconstruction implementation and experiments

> **Next-agent entry point:** this document is the single handoff; do not create separate handoff files. Read **Current state for the next agent** below, then continue the runbook in order. E00–E04 have been executed. E07's synthetic comparison has been run and adopted nothing. E10A's IR-dot stereo diagnostic is recorded; a live RGB+IR-dot TSDF refine is wired (`FRAME_IR_DOT=1`) but is not gate-G accepted. **E05 is closed as of 2026-11-06: gate not passed, recurrence stays off; E06 is next. IR yaw and dot-shift work is paused at the user's request (see D1).** **E05's off/on comparison ran on QNN on development data, and recurrence stays off by default. The warp and worker invalidation are now in C++; the wall dome is the priority: gated RGB plus IR improved walls offline, RGB bias is measured, desk improvement is unconfirmed, and tag-free yaw has not passed (see the E05 status).** Update **Current state** and the experiment's status block when you finish. The research assessment and architecture sections that follow explain the choices. This is a dedicated scan → finalize → save workflow; no game runs during reconstruction. The numerical compute budgets elsewhere are unmeasured targets.

## Terminology for this project

**Room meshing** is a useful name for the desired result: a triangle mesh of the room. For the entire scan → finalize → save process, **room reconstruction** or **scene reconstruction** is more precise. The mesh is the final geometric representation; several earlier stages determine its accuracy. If the wall is misplaced in depth or the camera pose is wrong, changing the triangulation alone cannot put it in the right place.

| Term | Meaning here | Typical output |
|---|---|---|
| Stereo depth estimation | Infer distance from a pair of camera images. The current model predicts disparity, which the calibrated rig converts to depth. | A depth or disparity image for one capture |
| Temporal depth filtering | Reconcile a new depth image with prior observations after accounting for headset motion and occlusion. | A less noisy depth image plus validity/reliability |
| Depth fusion / volumetric integration | Combine many posed depth observations into shared world geometry. The current candidate uses a truncated signed distance field (TSDF). | A persistent map or volume |
| Surface extraction / meshing | Convert the map or fused surface samples into vertices and triangles. Surface nets and dual contouring are extraction methods. | A triangle mesh |
| Mesh optimization / simplification | Adjust vertices or reduce triangles while protecting measured planes, corners, openings and thin objects. | A refined mesh |
| Room reconstruction | The complete process, including capture, depth estimation, poses, fusion, meshing, validation and saving. | A reusable room asset |

**Pose** means the camera's position and orientation for a capture. **Reintegration** means rebuilding or correcting fused geometry after a pose or depth estimate changes; merely moving the finished mesh does not correct inconsistent observations. **Confidence** is a quality signal attached to a depth observation; the model's auxiliary output should be calibrated before interpreting it as a probability or variance. **Completeness** measures how much real geometry was recovered, while **accuracy** measures how close reconstructed geometry is to its measured location. Neither alone establishes a good room mesh.

## Ordered experiment runbook

### Operating contract and result format

Work primarily in `/Users/owenelliott/Documents/projects/frame-spatial-runtime`; model/runtime investigation lives in `/Users/owenelliott/Documents/projects/spatial-stuff/depth-model`. Read applicable `AGENTS.md` files and the latest model handoffs first. Both repositories may contain ongoing work. Record commits **and hashes of relevant working-tree files**; preserve unrelated modifications. Source inspected during planning is not proof of the deployed build, backend, flags, available sensors or model timings.

Begin with **E00**, then proceed sequentially. An optional experiment can be recorded as skipped with its stated reason. A failed candidate is not the new baseline. Retain the last accepted configuration, proceed only when the next experiment's prerequisites hold, and record the decision. Fix invalid instrumentation/data before comparing algorithms. Do not port a failing quality candidate merely because accelerator resources are available.

Create a unique results directory, proposed as `experiments/room-reconstruction/<run-id>/`, in the runtime repository. Keep a manifest, machine-readable metrics, configuration, source hashes, logs and paired mesh/depth outputs. Persist seeds and input hashes. For each experiment record: `id`, `status`, `parent_baseline`, `dataset_split`, `changed_variables`, `parameters`, `backend`, `precision`, `quality_metrics`, `timing_p50_p95`, `peak_memory`, `failures`, `decision` and `next_experiment`. Distinguish GPU timestamp duration, CPU time, transfer/wait time, end-to-end latency and total finalize time. Never present a plan or skipped experiment as a result.

Use three data partitions: development for implementation, tuning for parameter selection, and held-out scenes for acceptance. Split by room/session, not adjacent frames. Freeze the tuning grid, selected configuration and tolerances before held-out evaluation. Compare algorithms on the same observations/poses/model outputs except in explicitly marked model or pose experiments. Use sequence-level summaries and repeat runs to assess variability; individual neighboring pixels/frames are not independent trials.

**Shared acceptance gate G:** compare mean and p95 geometry error, plane offset and normal error, boundary displacement, completeness, thin-object survival and ghost behavior. Jitter/triangle reductions cannot compensate for displaced geometry. At E02 establish measurement repeatability and freeze non-regression tolerances. Suggested starting tolerances are plane/boundary error increase no greater than the larger of 5 mm or twice measured reference repeatability, normal-error increase no greater than the larger of 0.5 degrees or twice angular repeatability, and completeness loss no greater than 2 percentage points. These are proposed project thresholds, not established sensor capability; if the reference cannot resolve them, obtain stronger reference data or explicitly limit the claim. Critical topology failures—closing a measured doorway, bridging separate surfaces, losing a required thin feature—fail regardless of averages. Report distance/view-specific failures rather than hiding them in a whole-room mean. A no-benefit change is rejected unless it has a separately measured performance/complexity advantage with equivalent quality.

### Current state for the next agent (update after every experiment)

*Update 2026-11-06: E05 closed with recurrence off; **E06 is next**, starting with its carried-over prerequisites. IR yaw and dot-shift work is paused at the user's request; findings are under D1. Do not resume it, or follow the older "Next" lists below for yaw, without asking the user.*

*Last updated 2026-11-04, E05 compared on QNN (recurrence off, see its status block), E10A active IR-dot live refinement verified on flat textureless walls, E07 synthetic recorded. Next: **clear-desk scoring, a held-out tagged session for tag-free yaw, RGB motion stereo, marker fix**.* Repository: `/Users/owenelliott/Documents/projects/frame-spatial-runtime`. Results live in `experiments/room-reconstruction/20261103T033058Z-e0{0,1,2,2b}/`, `20261103T073125Z-e03/`, `20261103T170832Z-e04/`, `20261103T171757Z-e05/`, `20261103T172714Z-e07/`, and `20261104T004533Z-e10a/`, each with a `README.md` and `manifest.json`. Calibration background: `docs/xrservice_passthrough_alignment.md`. The working tree is **uncommitted** on runtime HEAD `59a9521`. E04 added `depth_observation.hpp`, `depth_features.hpp/.cpp`, and `tools/test_depth_features.cpp`; it did not change fusion, the depth gate, or the live worker. E07 added `TsdfIntegrateOptions` and a per-voxel contradiction counter. The default arguments reproduce the previous update, and `tools/test_tsdf_volume.cpp` still passes. No option was turned on. **E10A live refine (opt-in):** `src/ir_pair_capture.cpp`, `src/ir_dot_bridge.cpp`, `src/ir_dot_refine.cpp`, `scripts/ir_dot_{lib,refine_bridge}.py`, hooks in `cloud_worker.cpp`; default off (`FRAME_IR_DOT` unset). Live viewer preferred path is `--depth-model-qnn` (~23 ms), not hybrid (~560 ms). Flat textureless surfaces: neural disparity prior gated at `unc < 200` (allowing 100-160 unc through), ZNCC score threshold relaxed to 0.35, inlier threshold 80 dots, 3D bounded plane projection flattening neural dome up to 0.65 m. The Frame's `~/frame-spatial-runtime` predates the 2026-11-04 IR rewrite; sync and rebuild before testing it. E03 scratch is `~/e03-probe/`; E02b scratch remains `~/e02-probe/`; E10A scratch is `~/e10a-probe/`.

**Backend policy (2026-11-04): the Python `hybrid` backend is strongly discouraged.** Do not build new experiments or live paths on it; use `prepared-qnn` (~23 ms against ~560 ms). The 2026-11-04 cross-check found QNN equivalent on real frames with zero and non-zero `hidden` (see E05 status), so E05 runs on QNN. The baseline below is not yet re-baselined. Keep hybrid for reproducing older results until `finalize` moves. `model_offline_batch.py`, `analysis/hybrid_ab.py` and the `FRAME_DEPTH_BACKEND` default still select it.

**Baseline "E02b"** (keep it unless the experiment changes a variable):

| Setting | Value / how |
|---|---|
| Model | hybrid backend, primary graph, package `~/frame-dsp/depth-model/depth-model-direct-v5`, recurrence **off** (zero hidden state, t=0). **Live viewer:** prefer `--depth-model-qnn` / `wave1-20261101` claim engine (~23 ms); hybrid is ~560 ms |
| Input | spherical 0.29°/px (`FRAME_DEPTH_SPHERICAL=1`) |
| Capture | **`FRAME_CAPTURE_TIMING=v4l2`**. Opt-in, so always set it. L/R slots share one camera sequence; pose at left end-of-readout − 8 ms (`FRAME_EXPOSURE_OFFSET_MS`) |
| Stereo calibration | medoid of XRService history (`FRAME_STEREO_CALIBRATION_POLICY=medoid`, default). Captures save `calibration-history-at-init.json` and `calibration-used.json` |
| Depth gate / weight | uncertainty < 89 plus 3-of-8 neighbour support; weight `(89−u)/(89−50)` clipped to [0.05, 1] |
| TSDF | 3 cm voxels, 12 cm truncation, max weight 32 |
| IR-dot refine | **off by default.** `FRAME_IR_DOT=1` integrates each IR depth map once at its own pose and corrects RGB on the fitted plane for 2 s. Needs a session `IR_YAW_DEG` (start-up refuses without one) and spherical input. See `…e10a/README.md` §7 |

**Workflow (avoids the fragile live DSP path):**
1. **Capture, camera only, no DSP; cameras stream only while the headset is worn:**
   `FRAME_CAPTURE_TIMING=v4l2 FRAME_DEPTH_SPHERICAL=1 [FRAME_SAVE_RAW=1] ./bin/capture_model_inputs DIR N SPACING_MS`
   This writes the rectified 616² pair, the spherical and pinhole model inputs of the same exposure, timing JSON and optional raw 2464² luma.
2. **Inference in one DSP session:**
   `~/frame-dsp/depth-model/.venv/bin/python scripts/model_offline_batch.py --package ~/frame-dsp/depth-model/depth-model-direct-v5 --out OUT DIR/frame-1-spherical …`
   This writes both disparity and both uncertainty planes plus `batch.json`. Output is bit-exact and matches live. Then run `cd ~/frame-dsp/depth-model/wave1-20261101 && ../.venv/bin/python frame/bench/chunk_map.py` and require `n_bad: 0`.
3. **Geometry:** the spherical conversion is `modelSphericalDepth` (C++), or the NumPy `pts()` in `…-e02/analysis/tagcmp.py`.
4. **Re-rectify a raw pair with another calibration:** `FRAME_STEREO_CALIBRATION=<json> ./bin/rectify_raw_pair raw-left raw-right prefix`.
5. **Tag reference (120 ±1 mm tag16h5, ~5 mm thick; IDs 0/2/3/4/5 on the wall):** `~/frame-spatial-components/build/frame_apriltag --raw X-rect-left.u8 --image-size 616 --tag-size-m 0.12`. PnP uncertainty is about ±11 mm in position and ~2.5° in normal at 1.2 m.
6. The live recorder (`FRAME_DEPTH_RECORD_V2`, `--depth-model --tsdf`) reopens the cDSP every frame and **failed on its 7th close**. Use it only when live behaviour is under test.

**Hazards:**
- **DSP stuck:** `sudo systemctl reboot`, then `sudo insmod ~/frame-dsp/kmod-7.2/cdsp_mem.ko size_mb=128` (kernel 7.2.0-g9a8c8ea46950; the wave1 kmod is the older kernel), then `sudo sysctl -w kernel.yama.ptrace_scope=0` (a reboot resets it), then chunk_map with `n_bad: 0`. `recover_resident_exit.py` writes nothing unless the VTCM lease and bootstrap match a completed exit; E10A hit (2,0,8) with a zero lease and had to reboot. QNN live sessions can leave the same (2,0,8).
- **USB after reboot:** Type-C can stay in gadget/`device` role so `lsusb` is empty. Force host: `echo host \| sudo tee /sys/devices/platform/soc@0/a600000.usb/usb_role/a600000.usb-role-switch/role`, then reseat BELICE. Expect `04d8:00dd`. Check before `FRAME_IR_DOT=1`.
- **Never stream from `/dev/video*`.** V4L2 use is read-only `QUERYBUF` on XRService FDs. Leave XRService's `xrservice_ioctl_probe.so` hook (from april-tag-xrlib) in place.
- Do not leave the BELICE-850 emitter on continuously. Pulse only. Aim at matte surfaces.
- `bin/test_occlusion_capture_generation` fails to link; this predates E02.

**Rules in force:** gate G as frozen in E02. Absolute depth is biased by stereo yaw drift (**D1**): judge geometry relative to tags and scene, and never report raw metric accuracy. The uncertainty head is **not** a variance and is not a reason to accept or reject geometry. E03 left the runtime gate unchanged. E04 left fusion unchanged: the CPU crease labels and one-sided normals failed the development tag check, and the declared radius and 15° angle were not retuned. E06 must not treat those labels or normals as accepted support. E07 did not adopt a TSDF component, and its oracle normals were analytic. `FRAME_DEPTH_TEMPORAL` stays **0**. E05 measured the invalidation contract and ran off/on on QNN on one development capture; it chose off. QNN sessions must stay short: 2.45 s exits cleanly and 17.9 s stuck. E06 was not started.

**Data (all development; no tuning or held-out scenes exist yet). New rooms and sessions need user captures with the timed workflow:**

| Path (`experiments/room-reconstruction/`) | Content | Pairing |
|---|---|---|
| `…e01/live-development/` | 12 stationary frames, textured room, model output | legacy (static) |
| `…e01/live-close-per-frame/` | door, wall, low-texture ceiling dome | legacy, slow motion |
| `…e01/live-bounded-1/` | black sensor noise | legacy |
| `…e02/live-static-tag/` + `static-model-out/` | 1 tag; spherical and pinhole output | legacy, static |
| `…e02/live-tag-distances/` | tag at 0.7 / 1.2 / 1.72 m; 5-tag wall | legacy, mostly static |
| `…e02/calibration-sweep/` | 2 raw pairs and 20 calibration entries | legacy (pair 2 mismatched) |
| `…e02b/tag-ab/ab-timed/` | 15 timed pairs of the 5-tag wall, head moving | **verified** |
| `…e03/model-out/` | spherical inference of the e02 distance sets (24 pairs, one DSP session) | legacy inputs; model outputs new |
| `…e04/` | synthetic edge scenes plus the E03 tag-interior transfer of the CPU feature reference | legacy inputs; no new inference |
| `…e05/` | history-invalidation contract on the e01 posed captures; QNN/hybrid cross-check in `crosscheck/`; QNN off/on and pull test in `qnn-ab/` | uses the capture below |
| `experiments/room-dataset/wall-prior-full-001/` (Frame only) | 90 s: tag-free wall, tags and controller trace, wall/bed edge, hand crossing; 5 Hz dots | **timed**; DSP was stuck during capture, see E05 status |
| `…e07/` | synthetic wall, sphere, corner, and 2 cm sheet; TSDF components compared with the model frozen | no capture and no inference |
| `…e10a/` | headset-mounted BELICE-850; IR pair rec-lit / motion d1–d4, e1–e4; controller wall traces; live IR-dot refine notes §7 | IR hybrid batch offline; live RGB QNN + CPU IR refine |

**E03 result (development only; baseline unchanged):** the raw uint8 score does not rank tag-plane error (tag interior Spearman −0.03, every bin 3–4 mm). The operating threshold keeps 100% of tag and dome pixels and 98% of a dark frame. Clipped-linear weights are slightly worse than uniform (mean absolute residual 30.2 mm versus 29.0 mm). A monotonic ceiling fit on this pool is flat. Block-match disagreements do rise above a score of 80, and that tail does not transfer to the tag plane. Plane 1 is the same positive disparity at column `u − d` (median agreement 0.078 px on texture). Occlusion is a separate class (score median 90); textured left/right disagreements are 0.3% of valid pixels and are not the large plane error. Full tables: `20261103T073125Z-e03/`.

**E04 result (development only; baseline unchanged):** the observation contract and the CPU reference are in the tree and are not in the fusion path. Synthetic paint, shadow, silhouette, roof crease, opening, predicted depth, luma-snap, and linear upsampling behave as specified (`20261103T170832Z-e04/synthetic.log`). On the same 6,668-pixel tag interior as E03, the 15° crease rule labels the face as a crease (800 of 894 static pixels; high-gradient false-geometric rate 0.888). Valid normals cover 10.5% of that interior and sit 13.6° median from tag PnP. A tag-center pixel's quadrants each fit to about 2 mm RMS and still differ by up to 22.5°. The tag border produces no depth silhouette. Full tables: `20261103T170832Z-e04/`.

**E07 result (synthetic only; baseline unchanged):** no component enters the live TSDF. Robust weighting leaves 1866 sphere vertices after the empty rounds. A per-frame weight cap of 1 matches the baseline at these distances. An analytic normal changes an oblique plane from 0.4 mm to 0.3 mm and is not eligible. Delaying a full free-space update until the third contradiction clears the sphere and still drops a 2 cm sheet from 3456 vertices to 0, the same as the baseline; the 0.5 cut was not lowered. The combination fails because of the robust weight. Full table: `20261103T172714Z-e07/`.

**Findings that were the E03 inputs:**
- Dark noise passes the gate: 96% valid, uncertainty p50 66–68. E03 remeasured 97.9% below 89 and 95.3% supported, p50 67.5.
- Low-texture domes are not flagged. E03: ceiling relative RMS 18–19%, score p50 50–51, 100% below 89, with the bulge more confident than the edges.
- The model matches 9×9 NCC block matching on texture (median −0.04 to −0.07 px); `…e02/analysis/bm.py`.
- CLAHE was not adopted.

**E10A result (development only; baseline unchanged):** headset-mounted BELICE-850 dots are usable as IR-pair stereo texture. Factory IR rig plus roll +0.4° and pitch 0.25° (vertical residual; stable). Yaw drifts per session and must be fitted: about −0.12° dark, −0.25° lit tags, −0.41° controller e-set. Controller trace plus stills at 1.5–2 m pin it (e1/e2/e3 zeros −0.410/−0.402/−0.456°). Dot period 3.45 px, unwrap, second search. Lit RGB-prior+unwrap offline: 83% kept, −28 mm median. **Live refine (opt-in):** RGB stays on `--depth-model-qnn` (~23 ms); a CPU IR+dot sidecar (~680 ms / 1.5 s) adds one posed IR observation per cycle and corrects RGB on the fitted plane. The code was rewritten 2026-11-04 after a review and has not yet run on the Frame. Env: `FRAME_IR_DOT=1`, session `IR_YAW_DEG`. Not gate-G; not the default baseline. Details and launch: `20261104T004533Z-e10a/README.md` §7.

**J01 offline junction diagnostic (2026-11-04; synthetic only; baseline unchanged):** `frame-spatial-runtime/experiments/room-reconstruction/20261104T163416Z-junction/` uses the current IR plane-fit/footprint/depth functions on exact wall-corner and tabletop geometry, then replays the resulting depth through the TSDF. Growing the footprint by one 10 cm cell corrected 2,376/7,209 wall-side pixels and 1,165/14,325 table-side pixels onto the wrong plane when side dots were absent. The grown meshes extended 50 mm past the wall corner and 70 mm past the table edge by triangle-centroid criteria; disabling growth produced no qualifying extension. Even 500 table-side dots did not block the error because the current footprint veto checks foreground dots only. This demonstrates a possible failure mechanism, not live prevalence or gate-G acceptance. `tools/mesh_viewer.py` compares saved PLYs side by side. E05 remains next; E09–E11 have not run.

**Dataset harness (2026-11-04; local smoke only):** `frame-spatial-runtime/scripts/room_dataset.py` and `docs/room_dataset_harness.md` record timed stereo NV12/luma/model inputs, paired raw IR, head/controller poses, thermal zones and dot pulses, then run one-session offline inference, tag detection, depth conversion and TSDF replay. `tools/mark_surfaces.cpp` adds a controller trace and headset overlay. The synthetic end-to-end fixture passed; the Frame was unreachable by SSH, so the capture tools and visualizer are not hardware-validated. This harness does not change E02b or complete E05.

**Open but not the next experiment's job:** probe-thread cost; refitting the 8 ms exposure offset (E02 item 2); D1 (IMX616 yaw); E10A floods-off during pulse, multi-frame pulse stack, gate G, and E09/E10 plane provenance.

### E00 — Freeze the real baseline and establish available execution paths

**Modelion:** what code, geometry convention and hardware path will the experiments actually use?

**Actions:** inventory runtime and model working trees, deployed binary/package hashes, backend selection, model resolution, primary/alternate graph, rectification/projection mode, disparity scale, temporal flags and mesh settings. Verify `--depth-model`, `--depth-model-prepared`, `--depth-model-qnn`, `--model-temporal`, `--model-spherical` and `--tsdf` against current source rather than copying historical launch strings. Record stereo/NPU resource ownership and the supported process for releasing it; do not assume a generic model compiler/runtime exists. Query Vulkan limits, storage formats, queue families, timestamp availability and whether preview/compute use the same device. Read the existing build instructions and align Makefile/CMake sources if needed.

**Artifact/gate:** a reproducible launch manifest and short smoke run with actual stage timings. Log rig intrinsics/extrinsics and projection units without redoing the user's calibration. If the baseline cannot run or units are unresolved, resolve that before E01. Current entry points: `src/model_depth_pipeline.cpp`, `src/cloud_worker.cpp`, `include/model_depth_geometry.hpp`, `include/tsdf_volume.hpp`, `scripts/model_depth_bridge.py`.

### E01 — Build a replay corpus and trustworthy geometric evaluator

**Modelion:** can we distinguish better reconstruction from a visually smoother error?

**Actions:** extend recording without breaking old formats; add versioned model outputs, uncertainty, validity, frame identity, capture timing/provenance, pose/tracking flags, calibration/config hashes and both images when required. Retain raw model-resolution observations, not only upsampled depth. Add an optional second-disparity output to the bridge with protocol versioning and shape checks. Reuse `scripts/model_temporal_replay.py` and `tools/test_tsdf_volume.cpp`; add a reconstruction replay entry point that can run without the live viewer. The current temporal replay exercises model execution and is not yet a complete map-quality evaluator.

Generate synthetic walls, oblique planes, non-right-angle corners, openings, thin structures and curved surfaces, with configurable depth bias/noise, motion and invalid samples. Capture real stationary/translated/rotating scans of independently measured planes, corners, doorways and thin objects; include moving occluders and texture/shadow edges. Record reference uncertainty. Do not freely ICP-align each output plane/region, because this can hide placement error; use one shared, documented reference transform and report alignment uncertainty.

**Gate:** replay reproduces the baseline and evaluator detects intentionally injected plane shifts, rounded corners and closed holes. Save the dataset manifest and split. If recording lacks sensor exposure timestamps, label the proxy and quantify its limits in E02.

### E02 — Separate timing, projection and resolution errors from fusion errors

**Change:** diagnostic evaluation only; retain the existing predictor and fusion.

**Actions:** compare stationary and moving scans against the same reference. Audit exposure versus buffer-read midpoint, pair synchronization, rectified camera-to-head transform, disparity conversion, model-resize pixel centers and projection mode. If current source offers planar versus spherical model inputs, test those as a separate controlled input-convention comparison, not bundled with filtering. Diagnose units/sign/axis mistakes before algorithm changes. Retain both native-resolution and expanded depth metrics. Run fixed input several times to establish output/timing repeatability and reference resolution; freeze gate G here.

**Decision:** if stationary geometry is sound but movement causes major drift/thickening, prioritize alignment fixes and repeat E02. If static depth has systematic bias, document it; smoothing cannot establish absolute accuracy. A timing/projection correction becomes a new named baseline and reruns the baseline corpus. Do not tune arbitrary pose offsets to the held-out scene.

**Status (2026-11-03): diagnostic complete; baseline not accepted.** Results are in `frame-spatial-runtime/experiments/room-reconstruction/20261103T033058Z-e02/`.

What is established:
- Model output is deterministic, and offline replay equals live bit-for-bit.
- The spherical convention and disparity units are verified against block matching. Spherical beats pinhole on identical exposures.
- The L/R sensors expose within 0.16 ms at 60 fps.

What blocks acceptance:
- The slot picker returns stale or mismatched L/R slots. One pair showed 2.8 px of error.
- Pose is taken ~25–70 ms after exposure.
- XRService's online stereo calibration wobbles by about ±1.1 px of disparity between entries.
- Low-texture surfaces dome (predictor error). The uncertainty gate passes dark noise.

Gate G is frozen, with normal tolerance limited to 5° by the tag reference. Local contrast enhancement gave only minor gains.

**E02b (2026-11-03): capture fix accepted** (`…-e02b/`). V4L2-sequence-paired slots, exposure-timed pose and a per-capture calibration snapshot are in, opt-in via `FRAME_CAPTURE_TIMING=v4l2`.
- 1193/1200 frames paired, with 0 mismatches and 0 torn copies. Pair age is 19 ms.
- Per-tag pair-to-pair disparity sd during head motion fell from up to 4.2 px to ≤ 0.66 px.

Still open:
- XRService's online stereo calibration drifts by about 3 px over hours; it is the dominant static error.
- Refit the 8 ms exposure offset.
- Measure probe cost.
- Run the Model model live in timed mode.
- Stereo calibration: medoid-of-history policy now default; drift is deferred as **D1** (see Deferred items). Metric claims are blocked until D1 is resolved.

**E03, E04, and the synthetic half of E07 have been run.** Baseline settings, workflow, data and hazards are in **Current state for the next agent** above. E05 is next. E06 has not been started.

### E03 — Test the uncertainty head and stereo consistency

**Baseline:** E02 accepted geometry/configuration; no new temporal smoothing.

**Actions:** measure error versus raw score, range, texture, boundary class and incidence. Compare current clipped-linear confidence, uniform weight, and a monotonic mapping fitted only on tuning scenes. Test left/right consistency only after confirming both output planes' geometric conventions and actual stereo visibility. Treat occluded/unmatchable pixels separately rather than marking every consistency failure as estimator noise. Avoid converting the head to a variance unless empirical calibration supports that interpretation.

**Gate/decision:** retain weighting/rejection only when error-versus-coverage improves under G across held-out scenes. Save reliability plots and selected mapping. If the score does not transfer, keep uniform/conservative weighting and residual evidence; do not block the rest of the runbook on a useful uncertainty head. This is an empirical experiment, not a claim supported by the model's tensor name.

**Status (2026-11-03): measured; weighting not accepted.** Results are in `frame-spatial-runtime/experiments/room-reconstruction/20261103T073125Z-e03/`. No runtime default changed. Parent baseline remains E02b.

There is no tuning scene and no held-out scene, so no mapping was selected and gate G was not applied. The error is the tag-plane residual after one per-frame median, on the legacy static and distance captures (the E02b timed wall has no spherical inputs and was not inferred). Static frames reproduce E02's +115 mm tag bias; that bias is removed before ranking and is not an E03 result.

What was measured:
- Tag interior, 6,668 pixels: median absolute residual 3.3 mm, score median 62, Spearman −0.03. Bins from 45 to 70 are all 3–4 mm.
- The wall annulus is worse (median 18 mm, p95 106 mm) and more confident (median 55). A threshold of 50 keeps 31% of the reference and is 7 mm worse than a random subset. The threshold of 89 keeps 100%.
- Clipped-linear weights: mean absolute residual 30.2 mm versus 29.0 mm uniform. An isotonic curve fit on even frames and scored on odd frames is flat.
- Block match on the textured room, 16,070 samples: Spearman +0.19, and +0.35 on strong texture. Scores of 80–89 and ≥89 have median disagreements of 0.98 px and 1.42 px, against 0.15 px overall. The gate at 89 changes the median by 0.001 px. A threshold of 70 cuts the p95 from 1.20 px to 0.73 px but keeps only 73% of matches, which fails the completeness rule, and that stricter cut hurts the tag plane.
- Dark frames: score p50 67.5, 97.9% below 89, 95.3% also passing 3-of-8 support.
- Ceiling dome: relative RMS 18–19%, score p50 50–51, 100% below 89. The centre bulge (+177 mm) is scored 46–47; the edges (−300 mm) are scored about 60.
- Left/right: plane 1 at `u − d` agrees with plane 0 to 0.078 px median on texture. `u + d` is 0.60 px; the negated samples are about 19 px. Of valid pixels, 73% are low texture (score 55, residual 23 mm), 24% agree within 1 px (score 61, residual 8.9 mm), 0.28% are textured disagreements (score 80, residual 10.8 mm), 0.22% are occluded (score 90) and 2.3% fall outside the other image (score 71). Occlusion and out-of-image are visibility, not estimator noise. The large plane error is the confident low-texture wall.

The score is not a variance: tag-interior error is flat across it. Uniform weight is the honest description of the operating gate on real geometry. Changing the code to uniform is not accepted either; it was not held out, and it would not repair the dome or the dark frames. Do not block E04. E07 must not treat this head as a variance.

Inference, one session, 24 distance pairs: first call 808 ms, then p50/p95 590/593 ms of Python wall clock around `infer` (bridge transfer included, not a GPU timestamp). `chunk_map` `n_bad: 0` before and after. No volume was fused.

### E04 — Establish edge classes and robust normals on CPU

**Actions:** add the shared observation contract and CPU reference feature implementation. Compare depth-only boundaries, luma-only candidates, and combined depth/normal/multiview validation. Use one-sided compatible neighborhoods for normal estimation. Include painted lines, shadows, depth silhouettes, actual creases and mixed-depth upsampling pixels. Preserve missing-data and provenance masks.

**Measurements/gate:** boundary precision/recall against labeled geometry, boundary location error, normal error and valid-normal coverage; no false geometric ridges on appearance-only edges. Save overlay/debug images. Reject a classifier that sharpens appearance while shifting silhouettes. Adopt the smallest useful feature set. [Sharp-feature research](https://people.csail.mit.edu/jiex/papers/SharpFeatures/paper.pdf).

**Status (2026-11-03): measured; no feature set accepted.** Results are in `frame-spatial-runtime/experiments/room-reconstruction/20261103T170832Z-e04/`. The new files are `include/depth_observation.hpp`, `include/depth_features.hpp`, `src/depth_features.cpp`, and `tools/test_depth_features.cpp`. Fusion, the depth gate, and the live worker are unchanged. Parent baseline remains E02b.

There is no tuning scene and no held-out scene, so gate G was not applied. The radius (5), crease angle (15°), luma gradient (24), and jump tolerance (max of 4 cm and 4% of z) were frozen before the tag frames were scored.

Synthetic scenes, all passing, are in `synthetic.log` and `overlays/synthetic/`. Luma-only marks every paint and shadow edge as a silhouette, and a luma snap moves a depth step from column 69.5 onto a decoy at 75. Depth jumps find that step and miss a 12°+12° roof. The combined estimator labels the roof, withholds the averaged normal on the crease, leaves paint as appearance, keeps predicted depth out of measured support, and widens a linearly upsampled step from 1.0 px to 1.5 px. A one-view false box loses all 196 silhouette pixels when the other view is required; the step present in both views keeps all 192. That confirmation is not the single-view default. CPU time on a flat 320×256 frame is p50/p95 66.0/67.6 ms of `steady_clock`, not a GPU timestamp.

The development transfer uses the E03 tag interior (6,668 pixels; far frames 2–4 still skipped). Static interior class counts are 800 crease, 76 appearance, 18 interior, and 0 silhouette. The high-gradient false-geometric rate is 0.888 against a predeclared 0.02. Normal coverage is 0.105 against 0.90, and the emitted normals have median error 13.6° against 5° (14.9% within 5°). Near, center, and far agree (crease shares 0.98, 0.80, 0.87). The tag border silhouette rate is 0, which matches a 5 mm step sitting inside the 4 cm tolerance. At the static frame-1 tag center, four open quadrants fit to 1.7–2.6 mm RMS and differ by up to 22.5°, so the crease rule is reading local tilt in the depth map. The wall annulus against the same PnP plane is 34.7° median on the static set and 11–13° on the distance sets; that annulus is the dome E03 already measured. Thresholds were not changed after these numbers.

No feature set enters fusion. E06 and E07 must not treat these crease labels or normals as accepted support. Continue to E05.

### E05 — Isolate the existing model's temporal recurrence

**Actions:** replay identical stereo pairs with `FRAME_DEPTH_TEMPORAL=0` versus `1`; keep downstream gates, TSDF and projection fixed. Verify the existing replay script's selected backend matches the comparison being claimed. Inject tracking loss, long gaps, fast rotation and capture reset to verify history invalidation. Preserve raw disparity and history-warp diagnostics.

**Gate:** improve pose-aligned jitter while meeting G and not increasing temporal lag/ghost lifetime beyond frozen tolerances. If recurrence fails, retain it disabled and continue. Repeated history agreement must not be treated as independent evidence. Output the chosen recurrence state explicitly; do not silently enable it for every later test. [Meta's consistency/correction tradeoff](https://openaccess.thecvf.com/content/CVPR2023/html/Khan_Temporally_Consistent_Online_Depth_Estimation_Using_Point-Based_Fusion_CVPR_2023_paper.html) is context, not validation of the recovered hidden-state contract.

**Status (2026-11-03): invalidation measured; the model comparison was not run.** `FRAME_DEPTH_TEMPORAL` stays 0. Results so far are in `frame-spatial-runtime/experiments/room-reconstruction/20261103T171757Z-e05/`. The Frame at `10.20.41.156` did not answer ARP, so no hybrid session was opened and there is no jitter number. Gate G was not applied. This does not finish E05.

`scripts/model_temporal_replay.py` is not the comparison. It opens a prepared QNN session and a pinhole `ModelTemporalState()`. The baseline is hybrid primary, spherical 0.29°/px. `analysis/hybrid_ab.py` is the one-session runner and has not been executed.

On the existing posed captures, a saved full-frame disparity is still history after every one of 22 steps. `inputs()` takes a pose and no timestamp. The longest gap, 1515 ms in `live-close-per-frame`, splats 0.957 of the half-resolution grid. `live-bounded-1` frame 1→2 is 51.7° and still splats 0.598. All 11 `live-development` steps fall under the graph's still gate (`mean |t|` at most 5.9e-4, threshold 7.5e-4), so hidden features would be scaled by 0.8 on every pair. A call with no motion sends zeros for that call and leaves the saved disparity in place; the next motion warps it. `head.valid == false` in `cloud_worker.cpp` returns before the bridge, leaves `previousModelAtCapture` set, and does not reset the Python state. `ModelMotion.reset` is never assigned. The bridge would honor bit 1. None of the 25 recorded frames set `motion_reset`.

**Update (2026-11-04, Frame): new capture and QNN cross-check; off/on comparison still not run.** `experiments/room-dataset/wall-prior-full-001/` on the Frame (17 GB, not copied to the Mac) is 90 s and 750 timed spherical frames at about 8.5 Hz, with 1,365 IR pairs, 414 fired pulses, 48 shams and 341 controller marks. Its segments are tag-free wall still, same wall moving, tags with a controller trace, and a wall/bed edge with a hand crossing. Head movement during the "still" segments means the stillest windows are only about 5 s (frames 124–163, 491–530, 613–653). Recorded with the DSP stuck at (2,0,8) from an earlier session, so XRService's cDSP calls were failing. Across frames 335–749, tag world positions agree to about 2–7 mm median and 10–27 mm p95 against ±11 mm PnP uncertainty. The tag-free frames before that have no such check, and the calibration snapshot's dependence on those failed calls is unknown.

`analysis/backend_crosscheck.py` ran 16 frames from four windows through hybrid and QNN on byte-identical inputs (0 input hash mismatches). Mean absolute disparity difference, worst frame: zero `hidden` 0.093 px, forced `hidden` 0.089 px, free-running 0.089 px. The effect of `hidden` matched in size (hybrid 0.08–0.30 px, QNN 0.08–0.29 px) with correlation 0.91–0.99. A QNN repeat pass later in the session was bit-exact. At most 1.13% of pixels differ by more than 0.5 px; the single worst pixel differs by 21 px and has not been located. Median time 511.6 ms against 23.5 ms. Both sessions closed with `chunk_map.py` at `n_bad: 0`; the QNN session lasted a few seconds, so this says nothing about the long-session exit fault. Results: `crosscheck/` in the E05 directory. Which backend is more correct is not established.

**Off/on comparison (2026-11-04, QNN, development only). Chosen recurrence state: off.** All 750 frames ran with `hidden = 0` and with recurrence, using `analysis/qnn_ab.py`, whose metrics were declared before the first run. Results: `qnn-ab/` in the E05 directory.

| Segment | Consecutive-frame median, off → on (px) | > 1 px, off → on | About 1 s horizon median, off → on (px) |
|---|---|---|---|
| Tag-free wall, stillest window | 0.205 → 0.097 | 3.3% → 2.4% | 0.320 → 0.240 |
| Tag-free wall, moving | 0.240 → 0.152 | 6.3% → 3.1% | 0.831 → 0.776 |
| Tags | 0.231 → 0.164 | 9.4% → 6.9% | 0.872 → 0.801 |
| Edge and hand | 0.091 → 0.057 | 4.9% → 4.3% | 0.175 → 0.148 |

This agreement is partly circular, because recurrence is fed the same warp. The independent measure, per-tag standard deviation of the plane offset across frames, moved only from 9.4 to 7.3 mm on the least-seen tag and by 0.2–1.9 mm on the other five, and it mostly reflects view changes rather than frame-to-frame jitter. Gate G on tag frames: mean |offset| change −1.6 mm (≤ 1.25 m, n=148) and −3.0 mm (1.25–2 m, n=405), p95 +8.5 and +6.7 mm against limits of 15 and 30 mm; normal change p95 +1.8° and +2.0° against 5°. Completeness loss is at most 0.0 points. Ghost: peak 0.28% of pixels and longest run 4 frames (about 0.5 s); lag: peak 0.17% and longest run 22 frames (about 2.6 s) above a near-zero level. No hand-shaped trace is visible after the hand leaves (`qnn-ab/ghost-sheet.png`). The ghost metric is whole-frame and does not isolate the hand. **E02 froze no numeric ghost or lag tolerance, so that part of the gate is reported and not passed.** The off run cannot show a ghost, so any nonzero value is strictly an increase.

The decision stays off because this is one development capture, the ghost tolerance does not exist, and the invalidation holes above are unfixed. Recurrence is a viable arm for the E06 matrix once the worker resets history on tracking loss, sets `ModelMotion.reset`, and adds time and rotation limits. Python-state probes: lost pose and reset reproduce the zero output exactly; correct-pose history from 42 s earlier changed no pixel by more than 0.84 px at 2.5% splat; the same history with no pose update changed 2.8% of pixels by more than 1 px, with a 2.49 px maximum. These do not exercise the worker path.

**`hidden` behaves as a local gated blend, not a neural prior.** Synthetic patches around the zero-state output moved the output by the gate value (about 0.8 or 0.5) times the offset, identically on low and high texture, and the same whether the patch or the whole frame was offset, with mean change 0.000 px in a 16 px ring outside the patch. ±8 px had no median effect. On the saved run, `output = off + g × (hidden − off)` inside a window holds on most pixels (97.6–97.8% within 0.1 px with one fixed window per gate; `analysis/hidden_rule.py`). The window grows with disparity: at gate 0.5, roughly −1.5/+2.5 px at d 8–11 and −3/+4 px at d 15–25, narrower at gate 0.8. This is an empirical description and not a graph-level proof. It means an IR plane fed through `hidden` could not spread across edges, would not favour low texture, and would do nothing outside the window, so a CPU blend would do the same job without the DSP. Do not pursue the IR prior through `hidden`.

**Exit fault:** a 17.9 s QNN session (750 frames) stuck at (2,0,8) after a 120 s exit wait, while 18 sessions of up to 2.45 s were clean: 14 on the recurrence run, 3 on the pull test and 1 on the cross-check. The threshold is therefore above 2.45 s and at most 17.9 s on this one sample, far below the previous 125 s observation. Repeated session close did not fail. `qnn_ab.py run` splits long sequences into 60-frame sessions and resumes after a reboot. Output did not depend on session pattern: pull-test baselines were bit-identical to the long zero run.

**C++ warp and first live run (2026-11-04).** `src/model_temporal.cpp` replaces the Python warp in the live path: the worker holds the state and sends `hidden` and `t` to the bridge (`--host-temporal`), and `tools/test_model_temporal.cpp` checks it against `scripts/model_temporal.py`. On 94 cases from real output the largest `hidden` difference was 0.000008 px with `t` identical, at 2.8 ms p50 for warp and save against about 7 ms. The worker now drops history on tracking loss, on a pose gap over 250 ms and on rotation over 30° per step; these limits are unmeasured starting values. `FRAME_DEPTH_TEMPORAL` still defaults to 0. A 3.5 min headset run (`--depth-model-qnn --model-temporal --tsdf`, log in `live-cpp-temporal/`) produced 2,503 depth frames at about 11.8 Hz, about 72 FPS rendered and 442,298 triangles, with history applied on 2,497 frames, 2 gap drops and 1 rotation drop. Logged warp 3.7 ms p50, inference 23.8 ms p50, no compositor crash. Tracking loss did not occur, so that path has only unit-test coverage. The exit stuck at (2,0,8) and needed a reboot, as expected for a long session. This run has no recurrence-off pair and saved no mesh, so it is not a quality result.

**Wall dome diagnostic (2026-11-04, recurrence off, 11 frames of the tag wall at about 1.3 m; `analysis/dome_source.py`, `wall-error/dome_source.txt`).** The user reports walls that pillow toward them, angled planes past the surface, holes, and a mesh that only lines up at sharp angles; this predates recurrence. Against the disparity expected from the tag plane, 9×9 NCC block matching is nearly flat with image angle (+0.49 px at 0–10°, +0.77 px at 40–55°), while the model domes (+1.38 px at 0–10°, −1.04 px at 40–55°). Where the block matcher is confident (1,733 of 15,199 wall samples, mostly textured pixels), the model stays close to it (+0.48 to −0.25 px). The median vertical offset is about −0.1 px at every angle, so rectification does not explain the dome. So there are two separate errors: a model dome on low-texture wall of roughly ±0.2 m at this range, and a roughly constant +0.6 px disparity bias consistent with D1 and the tag offsets of about 100 mm. These pixel counts favour textured areas and one wall, and the angled planes, holes and sharp-angle behaviour have not yet been traced to these errors. The viewing-angle tables from `analysis/wall_error.py` are invalid and were not kept: its "near wall" pixels include furniture, and the tag-free segment is a different wall. Its `wall-distance.png` is kept only for the tag-wall column.

**Wall mesh comparison (2026-11-05, offline, one take; `analysis/wall_meshes.py`, results in `wall-meshes/`).** Five TSDF meshes from the same take, with the gate and metrics declared before the first mesh. The gate keeps a model pixel only with ZNCC ≥ 0.6 at the model's disparity, at least 0.1 above ZNCC 3 px either side, and window std ≥ 2; it keeps 18.4% of pixels. IR yaw swept on the tag wall gave −0.30°: median plane offset +5 mm, plane tilt 0.77°, raw dot RMS about 10 mm. Tilt barely depended on yaw, so it is largely independent of the fit. 405 of 406 accepted lit pairs refined, at about 0.58 s wall time each using about 3 CPU cores. Tag wall scores:

| Variant | Area (m²) | Within 60 mm | In front > 60 mm | Behind > 60 mm | Tilted > 20° | 5 cm cells on the wall |
|---|---:|---:|---:|---:|---:|---:|
| rgb | 8.42 | 20.6% | 25.1% | 54.3% | 79.8% | 655 |
| gated | 0.84 | 42.5% | 57.5% | 0.0% | 70.9% | 192 |
| rgb+ir | 9.39 | 28.1% | 15.3% | 56.6% | 59.8% | 1086 |
| gated+ir | 4.38 | 76.5% | 19.2% | 4.2% | 30.8% | 1375 |
| gated-offset+ir | 4.42 | 78.3% | 16.5% | 5.2% | 24.6% | 1424 |

The extent is about 3.8 m² and 1,530 cells. Top views (`views-top.png`) show curved walls in rgb and straight walls with a clean corner in gated-offset+ir, including the tag-free wall. Adding IR without the gate does not fix the wall, because domed RGB still fuses. The bias correction adds little and is fitted on this wall. **Limits:** IR depth is a fitted plane, so flatness is by construction; yaw and the offset were fitted on the scored wall, so offsets are partly circular; this is one development session, not a gate G acceptance. The residual in-front and tilted area was not separated into hand, controller, bed, wall stair-stepping or real error. The live RGB correction was not reproduced. Junction overshoot (J01), non-planar objects and furniture coverage were not scored, and total triangles fell from 90,220 to 48,806. Refinement cost rules out 5 Hz live, but fits the scan → finalize workflow. A session yaw without tags is still needed.

**Tag-free yaw and desk take (2026-11-05; results in `motion-yaw/` and `desk-meshes/`; product decision: real users have the emitter, and no-emitter wall estimation is a later edge case).** `experiments/room-dataset/desk-full-001/` on the Frame is a 90 s take of a cluttered desk with no tags. The controller marker recorded 0 points because its actions were inactive (`actions_active=0000`), so it has no direct yaw reference. Plane-offset methods failed: spread across yaw was noisy, and the distance-squared fit (`yawfit`) gave −0.265° with a 0.5° range on the wall take and no zero crossing on the desk. Motion stereo on unlit IR corners (`analysis/motion_stereo_yaw.py`, 6,960 wall samples) failed on median alone (−0.223°). A joint yaw and scale fit (`analysis/motion_stereo_fit.py`, added after that failure) passed the wall rule narrowly: −0.253° (−0.281 to −0.224) against the tag's −0.30°. **It is not trusted yet:** the fitted scale disagrees between sessions (wall +0.5%, range −0.08 to +1.04; desk −2.1%, range −2.6 to −1.6), the desk gives −0.095° jointly against −0.193° on median alone, the per-degree slope differs from theory (6.32 and 6.61 against 6.90), and segment yaw varies by up to about 0.1° within a take.

Desk meshes used a provisional IR yaw of −0.30°, unmeasured for that session. **The wall fix does not transfer to the textured desktop.** On a central desk patch, both gated meshes keep a layer about 120–130 mm above the IR plane estimate, and gated+ir has two layers: 25th percentile −112 mm, median −27 mm, 75th percentile +120 mm. Only 7% of area is within 30 mm of the IR plane estimate and 60% is tilted, about the same as rgb. The most likely cause is the RGB disparity bias on textured pixels, which the gate keeps, against the IR plane; this is not yet proven, and IR yaw is unknown here and the patch contains objects. IR plane height above the floor plane is about 0.7 m and needs a tape measurement. Edge overshoot and object survival remain unscored.

**Tagged desk take (2026-11-05; `experiments/room-dataset/desk-tags-full-001/` on the Frame, results in `desk-tags-meshes/`, `analysis/desk_truth.py`, `analysis/desk_sections.py`).** Tags 1 and 2 on the wall and tag 3 on the desktop; world positions agree to 4–7 mm median across views, and wall and desk normals are 89.2° apart. IR yaw against tags: about −0.35° on the wall and −0.30° on the desk, so −0.35° by the declared rule (refined 427 pairs). RGB bias on gated pixels: +0.85 px on the desk (+0.83 and +0.78 px at disparity 9–13 and 13–20) and +1.03 px on the wall, rising to +2.6 px at disparity 3–6, which suggests some domed pixels pass the gate. Depth from tag PnP uses no stereo.

| Mesh | Wall within 30 mm | Wall within 60 mm | Wall cells | Desk disc within 30 mm | Desk disc within 60 mm |
|---|---:|---:|---:|---:|---:|
| rgb | 9.2% | 23.9% | 347 | 6.1% | 35.5% |
| gated | 0.5% | 5.7% | 38 | 6.5% | 37.0% |
| gated-offset (0.85 px removed) | 32.0% | 49.2% | 223 | 35.5% | 45.0% |
| gated+ir | 49.3% | 61.2% | 811 | 17.6% | 32.5% |
| gated-offset+ir | 59.1% | 64.9% | 826 | 26.1% | 34.3% |

**The declared desk rule failed (26.1% against over 50%).** The 0.7 m disc includes keyboard, stand, clutter and possibly the front face, so it understates a real result, but no valid clear-desk score exists yet: a ring around the tag also reaches past the front edge (median −48 mm and 40.2% within 30 mm for gated-offset+ir; +39 mm and 6.0% for rgb). **Correction to the previous block:** a section through the tag shows one thin desktop line in every mesh, not two layers. The split quartiles on the untagged take may come from clutter and its provisional yaw, so the double-layer explanation is not established. RGB bias itself is real and large. Combining bias removal with IR gave no gain over either alone near the tag, and its median moved below the tag plane, so a remaining conflict between them is possible. Tag-free yaw on this take: joint fit −0.139°, median −0.255°, against −0.35°, so both fail and the joint scale fit is dropped. Median-only is high by +0.077° (wall take) and +0.095° (this take), which suggests a fixed correction; this is two sessions, found after the fact. Controller marker was inactive again (`actions_active=0000`) on both post-reboot takes, while it worked before the reboots.

**Soft weighting, IR-plane bias and clear desk (2026-11-05; `analysis/soft_mesh.py`, on the tagged desk take).** *RGB bias from IR planes passes:* +0.80 px (0.76–0.83) against +0.85 px from tags, inside the 0.15 px limit, so RGB bias needs no tags once IR yaw is known. It also gives +0.66 px on the wall take (+0.615 by block matching against tags, a different measure) and +0.57 px on the untagged desk at its provisional yaw. An IR yaw error of 0.05° is about 0.17 px, so it carries directly into this estimate. *Soft weighting is a trade-off, with no clear win:*

| Unsupported RGB weight, with bias removed and IR | Wall within 30 mm | Wall within 60 mm | Keeps bias-corrected surface | Triangles |
|---|---:|---:|---:|---:|
| 0 (`gated-offset+ir`) | 59.1% | 64.9% | 75.4% | 52,528 |
| 0.02 | 56.9% | 62.6% | 77.8% | 54,206 |
| 0.1 | 50.1% | 55.5% | 85.0% | 58,122 |
| 0.3 | 42.7% | 51.5% | 93.5% | 63,956 |
| 1 | 32.9% | 43.2% | 100% | 76,228 |

Surface kept is measured against the weight-1 mesh with 5 cm voxels dilated by one, because the first comparison against uncorrected `rgb` undercounted corrected meshes. The monitor itself was not scored. *Clear desk:* 0.44 m² of cells; share with a surface within 30 mm of the tag plane: rgb and gated 0%, rgb+ir 48.4%, gated+ir 54.6%, bias-corrected variants 96–97%. **The bias-corrected figure is partly by construction, because those cells were selected with bias-corrected RGB.** The IR-only gains are not selected that way, though the unsupported weight and yaw still come from this session. Area and tilt on this mask are not meaningful: mesh area in the cells is about 2–3 times the cell area, so the columns also include surfaces above and below the desk.

Next (superseded by the closure below): visual check of the soft meshes; test the median-only IR yaw correction on a third tagged session held out from the correction; fix the marker. The recurrence off/on mesh comparison is lower priority, then E06. Run `chunk_map.py` before any capture as well as before inference, and require `n_bad: 0`.

**Closure (2026-11-06, user approved): E05 closed; gate not passed; chosen recurrence state: off.** `FRAME_DEPTH_TEMPORAL` stays 0. Jitter improved, partly circularly, and the tag frames stayed within gate G limits on one development capture. The lag and ghost part of the gate was reported but not passed, because no numeric tolerance was frozen. This follows the rule "if recurrence fails, retain it disabled and continue"; it does not show recurrence is harmful, and it remains an arm of the E06 matrix. Carried to E06: freeze ghost and lag tolerances, exercise tracking loss on the real worker, and compare recurrence off/on in meshes.

The wall dome, IR yaw, gating and soft-weighting work above was recorded here but is not part of E05's gate and does not block closure. Later corrections to it, including the tag yaw of −0.30° and the fixed median-only correction, are under D1.

### E06 — Test external pose-aware temporal filtering

**Actions:** implement bounded inverse-depth history with visibility resolution, robust innovation gating, edge-compatible support and expiration. Start with a single surface hypothesis plus explicit conflict handling; add a second hypothesis only if boundary tests justify it. Evaluate the full 2×2 matrix of recurrence off/on and external filter off/on, using the same TSDF. Tune a small declared grid of residual thresholds/history lifetime on tuning scenes. Test new foreground, disocclusion, moving people, holes and stale poses.

**Gate/decision:** accept under G with lower jitter and no added latency/ghost artifacts. If no combined benefit exists, retain only the winning simpler temporal mechanism. Save warped-depth/conflict visualizations. [Gaussian-mixture temporal depth fusion](https://ntrs.nasa.gov/citations/20210007922) motivates hypothesis handling; this implementation is not a reproduced paper until verified against its full algorithm.

**Prerequisites carried from E05 (2026-11-06):** freeze numeric ghost and lag tolerances before running the matrix, since E05 could not pass that part without them. Exercise tracking loss on the real worker path, which so far has unit-test coverage only. Include recurrence off/on mesh comparison in the matrix.

### E07 — Test robust projective TSDF updates

**Actions:** keep voxel size/truncation fixed initially; compare baseline versus residual-robust weights, evidence caps, valid-normal incidence weighting and conservative free-space contradiction. Add each component separately before evaluating the combination. Missing depth never carves. Require repeated valid contradictions for durable removal. Keep model/temporal output frozen so fusion quality is isolated. Record updated voxel counts and memory.

**Gate:** G plus removal of transient foreground without wall holes or disappearance of thin static objects. If the combination fails, identify the component causing it rather than retuning all parameters at once. Extend the current wall/sphere/removal test with corners and opposing observations of thin surfaces. [RoutedFusion](https://openaccess.thecvf.com/content_CVPR_2020/papers/Weder_RoutedFusion_Learning_Real-Time_Depth_Map_Fusion_CVPR_2020_paper.pdf) documents limitations of simple averaging; it does not establish this proposed weighting rule.

**Status (2026-11-03): synthetic comparison recorded; nothing adopted.** Results are in `frame-spatial-runtime/experiments/room-reconstruction/20261103T172714Z-e07/`. Gate G was not applied. The live defaults stay `robust` off, frame cap off, normal incidence off, and `carveAfter` 1. `tools/test_tsdf_volume.cpp` still passes.

The robust weight is `(1 - min(|s|, 1))^2`. It trims the noisy wall from 3.5 mm to 3.2 mm mean and leaves 1866 sphere vertices after 12 rounds of four empty poses, because a sample at the truncation edge has weight 0. The combination keeps those 1866 vertices for the same reason. A cap of `w <= 1` matches the baseline (723 leaves, sphere removed). Analytic incidence moves the oblique plane from 0.4 mm to 0.3 mm. Those normals are not the E04 estimator, and E04 was not accepted.

`carveAfter = 3` holds the front of the sphere for one empty view (5280 to 5280; the baseline goes 5256 to 5184) and removes it after the twelve rounds, with wall error 0.17 mm. One side of a 2 cm sheet produces 3456 vertices. The opposite view removes all of them for both the baseline and the delayed carve. The truncation is 12 cm, so each view's band covers the other face, and that residual is below the 0.5 delay. The cut was not lowered. A corner already sits at 0.019 mm under the baseline, and a missing-depth hole does not carve the wall (3728 triangles to 4710).

Integrate-only time on the 200² frames, steady_clock, 80 calls: p50 0.300 ms, p95 0.364 ms. The removal volume is 723 leaves, 388 bytes of payload each. That is not process RSS. E05 remains the next experiment. E06 was not started from this directory.

### E08 — Compare non-projective distance and volume resolution

**Actions:** compare a Voxfield-inspired normal-aware update against accepted E07 projective integration, using the same observations, support thresholds and mesher. First hold resolution fixed. Then test a bounded voxel/truncation grid—e.g. 2/3/4 cm voxels and 2/4/6-voxel truncation half-widths—as an explicit quality/memory/runtime tradeoff, not an expectation that the smallest setting wins. Exclude configurations outside measured memory limits. Do not interpret unreliable normals as exact geometry.

**Gate/decision:** retain non-projective integration only if G and oblique-surface accuracy improve at acceptable cost. Otherwise retain projective integration and continue. Freeze one configuration for plane experiments. [Voxfield paper](https://lucabartolomei.github.io/publications/IROS_2022_Voxfield.pdf).

### E09 — Detect and associate bounded plane patches without moving geometry

**Actions:** implement robust patch fitting, normal/offset uncertainty, spatial footprints and persistent association. Keep the TSDF and mesh unmodified. Track tentative/confirmed/contested/retired states; count independent viewpoints rather than duplicated/recurrent frame support. Test parallel offset planes, disconnected coplanar patches, doorways, curved objects and small protrusions.

**Gate:** stable patch identity, accurate normal/offset, correct observed footprint and no merging across openings/disconnected regions. Save patch IDs, support and residual maps. If reliable plane association is unavailable, do not proceed to pinning; use unconstrained TSDF while fixing E09. [PlanarMesh](https://arxiv.org/html/2510.13599v1) informs the representation; its LiDAR implementation is not assumed to work unchanged on stereo.

### E10 — Test adjustable plane pinning and correction recovery

**Actions:** compare unconstrained geometry, bounded soft constraints and a hard-snap diagnostic. Optimize against retained actual samples with robust loss; hold tracking poses fixed. Weight constraints using support/uncertainty and exclude unsupported boundaries. Preserve the raw measured volume. Deliberately introduce an early misplaced plane or bad first frame, then provide correct independent observations; test weakening, splitting and retiring constraints.

**Gate:** G, reduced flat-region jitter and successful recovery from wrong initial anchors. Reject a hard snap that looks flat but moves the wall or prevents recovery. Recompute constrained output from raw geometry when a plane changes; never count its own predictions as supporting evidence. [Plane-based optimization](https://arxiv.org/abs/1905.09829).

### E10A — Add active IR-dot sparse depth as independent plane evidence (conditional)

**Status (2026-11-04):** diagnostic captured out of order because the emitter was on hand. IR stereo and session yaw are in `experiments/room-reconstruction/20261104T004533Z-e10a/`. A live RGB+IR-dot TSDF refine is implemented and smoke-tested (`FRAME_IR_DOT=1`; README §7). Wall bulging fix implemented: RANSAC plane fitting on IR dots with a plane-aligned footprint grid, world-plane reprojection into later RGB frames, and one-time IR observations at the IR pose. There is no confidence suppression and no built-in yaw, despite earlier notes here. `test_ir_dot_plane`, `test_ir_dot_bridge` and `tools/test_ir_dot_lib.py` pass; it has not yet run on the Frame. Formal actions (a)(b)(c) and gate G below remain open. The ordered next experiment is still E05. Emitter is headset-mounted (per-frame stereo texture), not room-mounted; the room-mount rule below applied to tracking dots over time.

**Condition:** run only once a room-mounted IR dot emitter is available, E02b's exposure-timed capture is accepted, and E09/E10 show low-texture planes still failing G. E02 found that the low-texture ceiling/wall dome is predictor error: 18% relative plane RMS, which local contrast enhancement reduced only to 17.9%. Skip with that reason if E10 already meets G on low-texture planes.

**Why sparse depth, not texture:** the IMX616 colour cameras do not see IR. Synthetic dots painted into the luma images would carry no parallax. Pattern copies at the same pixel positions would read as zero disparity, and placing them at a chosen disparity presupposes the depth. Only the projected light, observed by IR-sensitive cameras, yields independent measurements.

**Prerequisites:**
- Read-only access to the OG01A1B tracking pair. E02 found it on XRService `/dev/video9` and `/dev/video13`: 1056×1024 GREY, 60 fps, kernel end-of-readout timestamps within ~1 ms of the colour pair. Pixel access needs an approach like the opt-in XRService `DQBUF` hook in `april-tag-xrlib`.
- The tracking-camera calibration from XRService.
- A rectification of that pair validated with the 120 mm tags.
- **Mount the emitter on the room, not the headset**, so the dots stay fixed to surfaces across frames.

**Actions:**
1. **Visibility and safety check:** confirm the dots appear in the tracking cameras and not in the IMX616. Measure headset tracking quality with the emitter on versus off; abort if tracking degrades.
2. **Dot depth:** detect dots and triangulate them in the IR stereo pair. Check their depth against tag PnP at several distances and image positions.
3. **Use the dots three ways, on identical observations:**
   - (a) as additional TSDF observations with their own weight;
   - (b) as independent support and residual evidence in E09 plane association and E10 soft constraints;
   - (c) as anchors for a bounded per-patch correction of the luma depth on confirmed planes.
4. **Provenance:** keep dot and luma observations separately tagged. Never count corrected luma depth as additional support.

**Gate:** G. On held-out scenes, low-texture plane offset, normal and flatness must improve without moving textured geometry or boundaries. Dot coverage gaps must not create false planes, and tracking must show no regression. If dot depth disagrees with the tags beyond G, fix the IR calibration chain before any fusion. If no held-out benefit appears, drop the emitter path and keep E10's result.

### E11 — Add crease constraints and boundary-protected simplification

**Actions:** derive finite crease segments only where supported plane footprints meet. Preserve occlusion outlines, openings and separate normal groups. Compare identical pinned geometry with/without boundary constraints, then simplify using a bounded position/boundary error. Include non-right-angle junctions, acute corners, doorframes, shelves and curved geometry. Do not add a Manhattan-world assumption implicitly.

**Gate:** G plus retained openings/thin detail and fewer planar-interior triangles. Record triangle count at matched error, not just a target decimation ratio. The earlier 70% planar-interior reduction is an aspirational goal, not permission to exceed geometric tolerance. [Sharp-feature extraction](https://people.csail.mit.edu/jiex/papers/SharpFeatures/paper.pdf), [plane-based optimization](https://arxiv.org/abs/1905.09829).

### E12 — Test incremental meshing and feature-preserving extraction separately

**Actions:** first compare full baseline extraction with dirty-block/neighbor-halo extraction using the same surface-nets rules; verify equivalence and seam ownership. Then compare constrained surface nets against a CPU reference for feature-preserving dual contouring. Exercise ambiguous cells, sparse observations, ill-conditioned normals, multiple features, capacity limits and eviction. Inspect cracks, winding, degeneracies and topology. Retain a fallback for unsupported cells.

**Gate/decision:** incremental extraction preserves geometry while reducing changed-region cost; accept the alternate extractor only if G and sharp-feature error improve. A mesher cannot recover a feature already erased by fusion. If dual contouring does not improve current stereo-derived fields, retain constrained surface nets. [Dual Contouring of Signed Distance Data](https://arxiv.org/abs/2604.00157) is a sampled-SDF reference, not a guarantee on noisy projective TSDF.

### E13 — Benchmark learned refiners on recorded scans before accelerator porting

**Order:** (a) Meta image-space fusion as the first external-depth baseline; (b) RoutedFusion's routing stage only if its depth/noise handling addresses a remaining failure; (c) V-FUSE for whole-scan multiview refinement. Run these sequentially on a suitable available desktop environment. If none exists, record this experiment as blocked by evaluation hardware and continue the geometric path; do not invent quality results. Inspect model weights, input contracts and licenses before integration. A routing-stage-only adaptation cannot claim the full RoutedFusion system's results.

Use the same keyframe/pose corpus and preserve required RGB if available; replicated luma is a separate domain-shift experiment, not an equivalent pretrained input. Compare original observations, accepted geometric filtering, learned-only refinement and their justified combination. For each refined result rebuild a fresh volume rather than adding corrected versions as independent samples. Account for selection/preprocessing and reconstruction time.

**Gate:** G and a repeatable held-out accuracy/completeness improvement sufficient to justify deployment complexity. Reject texture-copied detail, unsupported geometry and improvements confined to tuning scenes. If none wins, explicitly skip E17 and keep CPU/GPU geometry finalization. Sources: [Meta](https://openaccess.thecvf.com/content/CVPR2023/html/Khan_Temporally_Consistent_Online_Depth_Estimation_Using_Point-Based_Fusion_CVPR_2023_paper.html), [RoutedFusion](https://arxiv.org/abs/2001.04388), [V-FUSE](https://openaccess.thecvf.com/content/ICCV2023/html/Burgdorfer_V-FUSE_Volumetric_Depth_Map_Fusion_with_Long-Range_Constraints_ICCV_2023_paper.html).

### E14 — Port accepted dense features and temporal filtering to Vulkan

**Actions:** freeze the CPU reference and port E04/E06 in separate substeps. Use fixed layouts, capability-checked formats/atomics, deterministic nearest-surface selection and bounded buffers. Batch upload, feature extraction and filtering; avoid readback between stages. Keep original measurements available for diagnostics. Test CPU/GPU differences at silhouettes, invalid samples and occlusion conflicts, not merely whole-frame means.

**Gate:** gate G versus CPU output plus measured reduction in total processing time or CPU load without unacceptable power/preview regressions. Set numerical parity tolerances from reference precision before comparison. Measure upload, GPU passes, waits and wall time independently. Reject a faster kernel whose transfers make the application slower. [Khronos synchronization](https://docs.vulkan.org/guide/latest/synchronization.html).

### E15 — Port active TSDF updates and compact plane statistics

**Actions:** introduce pooled contiguous voxel blocks with block indirection; initially batch CPU allocation rather than building a fully GPU-allocated hash map. Update active blocks on GPU and reduce plane sufficient statistics to compact CPU summaries. Do not copy the whole map per depth frame. Compare integration and free-space behavior against E07/E08, including maximum capacity and block eviction. Audit FP16 storage versus FP32 accumulation and metadata overhead.

**Gate:** G and stable CPU/GPU integration parity, bounded memory, useful end-to-end speedup under a sustained scan. If allocation/control dominates, profile it before attempting GPU hash insertion. A CUDA mapping library is an architectural reference, not an installable Vulkan backend. [nvblox paper](https://arxiv.org/abs/2311.00626), [CUDA implementation](https://github.com/nvidia-isaac/nvblox).

### E16 — Port dirty-region extraction and GPU preview publication

**Actions:** port the accepted E12 extractor, output compaction, local plane projection and supported boundary constraints. Use capacity-checked prefix sums/counters and stable block-boundary rules. Query whether viewer and compute share `VkDevice`; use explicit supported interop or a measured staging fallback otherwise. Double/version buffers so rendering consumes only completed generations. Keep sparse CPU topology repair out of every-frame dispatches.

**Gate:** G plus no seams, overflow corruption or stale generation publication; measure mesh age, update frequency, missed display frames and GPU/CPU memory. Slice work to preserve a responsive scan preview. Async queues are an experiment, not an assumption of hardware overlap. [Khronos async-compute sample](https://docs.vulkan.org/samples/latest/samples/performance/async_compute/README.html).

### E17 — Deploy a successful learned subgraph to the NPU/DSP

**Prerequisite:** E13 identifies an actual quality winner. If not, mark skipped. A new distilled network is a separate training experiment requiring datasets, supervision, scene-disjoint validation and an explicit reason; it is not the default response to a difficult port.

**Actions:** inventory the selected neural subgraph's operators, shapes, normalization, precision and activations. First prove export/reference parity, then compiler/backend feasibility and small supported test execution. The recovered stereo executor is not automatically an arbitrary-model loader. Test supported higher precision before INT8 if available; calibrate on tuning scans and validate metric/boundary error on held-out scenes. Keep geometric reprojection, sparse fusion and topology outside the neural subgraph.

Drain stereo work, checkpoint, safely unload/release its resources and load the refiner sequentially. Measure graph switching, scratch/weights, transfers, inference, GPU preprocessing, reintegration, preview responsiveness and total finalize time; repeat switches and cancellation to verify cleanup and stereo restart. **Gate:** maintain the E13 quality gain and justify the port through total latency, power or GPU availability. If unsupported operators/backend work outweigh benefit or quantization destroys accuracy, use GPU refinement or omit the learned step. There is no preapproved NPU timing target or speedup claim.

### E18 — Compare whole-scan finalization strategies

**Actions:** compare (A) accepted capture geometry with plane/crease finalization; (B) selected keyframe geometric multiview refinement plus fresh fusion; (C) E13/E17 learned refinement plus fresh fusion, only if accepted. Use identical keyframe sets where possible; then explicitly compare coverage-based versus uniform temporal selection. Allow information from later observations at this stage, while preserving dynamic-object visibility rules. Keep original observations and provenance. Do not rerun a costly stereo model without a specific identified input/backend hypothesis.

Follow with bounded joint plane/crease refinement, affected-region remeshing, topology validation and error-constrained simplification. Save a preview checkpoint first and test cancellation/failure. **Gate:** choose the lowest-cost strategy meeting G and the desired saved-mesh quality. Report time-to-usable-preview separately from total finish time. Seconds-to-tens-of-seconds elsewhere are an allowance to investigate, not a promised outcome. Rebuild cost and quality determine the accepted method.

### E19 — Validate saved assets, collision geometry and relocalization

**Actions:** implement versioned indexed render and collision meshes with units, winding, normals, calibration/config/source identifiers, observed/unknown boundaries and persistent anchor metadata. Use the platform's actual supported localization mechanism. If a persistent-anchor API is unavailable, implement or explicitly specify a verified registration procedure before claiming cross-session placement. Simplify collision geometry with opening, boundary and displacement limits; test thin obstacles and gaps independently of visual mesh quality. Write atomically and preserve the previous valid asset.

**Gate:** save/reload identity within serialization tolerance; new-session relocalization within frozen placement tolerances; correct scale/axes and collision behavior in a small consumer harness. A mesh saved in current stage coordinates alone fails cross-session acceptance. Test partial/corrupt files, version mismatch and aborted finalization. This asset lifecycle is application engineering, not a paper result.

### E20 — Run the complete dedicated scanner and publish the handoff

**Actions:** run multiple held-out room scans and sustained capture sessions, initially at least 10 minutes or until thermal behavior stabilizes, extending if it does not. Include stop/resume, tracking loss, capture disconnect, map reset, memory pressure and finalization cancellation. Measure full process memory, inference throughput, GPU/CPU stage p50/p95, power/thermal indicators where accessible, preview/display misses, mesh age, finalize time and saved/reloaded geometry.

Repeat the original baseline and final accepted configuration on the same inputs. Publish a quality/runtime/memory comparison, known failure scenes, selected flags, backend/package/source hashes and exact reproduction instructions. Treat the 5–20 ms depth-processing and 128–384 MiB reconstruction figures as planning envelopes to compare against, not acceptance facts. If exceeded, reduce supported workload or improve measured bottlenecks while retaining G; document the actual achieved envelope.

**Handoff:** update this document with every E00–E20 status, links to result artifacts, rejected hypotheses and remaining blockers. Name the next unexecuted experiment and its prerequisites. Mark the system complete only when capture, finalization, persistence and verified reload/relocalization work together. Keep the last accepted configuration and a clean rollback path.

### Decision checkpoints

- **After E06:** establish whether model recurrence, external filtering or both help; retain only justified temporal complexity.
- **After E08:** choose the fusion formulation and resolution before testing plane constraints.
- **After E12:** the CPU reference should already demonstrate the desired plane/edge quality; GPU porting targets accepted behavior.
- **After E13:** NPU deployment is conditional on a learned refiner beating the geometric path on held-out geometry.
- **After E16:** determine whether GPU work improves whole-scanner cost and preview responsiveness, rather than only isolated kernels.
- **After E18:** choose the finalization quality/time tradeoff; more compute is useful only when the saved asset measurably improves.

### Deferred items (tracked, must be resolved before claiming metric accuracy)

**D1 — Stereo yaw drift: metric depth is biased.** Recorded 2026-11-03 and deliberately deferred by the user: "middle path".

- **Problem:** the IMX616 pair's rectified yaw is not stable. Against 120 mm tags, the yaw that makes stereo exact rose about 0.35° within 15 minutes of a cold boot (21.62° → 22.01°, XRService Euler component 0). Disparity error ≈ −7.4 px/° × (yaw used − true yaw), and relative depth error ≈ δpx × z / 24.3 px·m. That is about 11 cm at 1 m, 44 cm at 2 m and ~1 m at 3 m for the 2.7 px observed. This is 10–30× the model's stationary noise. Filtering, fusion and plane fitting cannot remove it.
- **Why XRService does not fix it:**
  - Its stereo estimator is rotation-only and image-match-based. Yaw is degenerate with depth there, so its estimates scatter ±0.15° (±1.1 px).
  - It applies stereo only at startup and only on an agreeing cluster; no `stereo:1` since 2026-11-02 22:06.
  - Model instead uses a whole-rig online calibration ("ocal": high-frequency → session → factory), looked up per depth timestamp and temperature-aware.
  - Details: `frame-spatial-runtime/docs/xrservice_passthrough_alignment.md`.
- **Done now (middle path):**
  - The stereo extrinsic is the **medoid** of XRService's history (`FRAME_STEREO_CALIBRATION_POLICY=medoid`, default; `latest` reproduces the old behaviour). This removes the newest-entry lottery: average 0.47 → 0.08 px against the consensus.
  - Every capture records its calibration snapshot, policy and chosen entry.
  - This does **not** track drift. The consensus can itself be off by 0.5–2.7 px from the tag truth.
- **Consequences until resolved:**
  - Absolute placement, cross-session alignment, and multi-distance consistency (wall doubling when walking toward/away) are biased.
  - Experiments E03–E12 must judge geometry **relative** to the tags or the scene (plane flatness, angles, boundaries, completeness), or use per-capture tag-fitted yaw for absolute claims. Never claim metric accuracy from the raw pipeline.
  - E19 (saved assets/relocalization) and E20 (final acceptance) **cannot pass** without D1.
- **IR pair findings (2026-11-06; paused at the user's request, not resolved).** Offline except where noted, development evidence only, no defaults changed. Results are in `priv-experiments/room-reconstruction/` (ignored by git) in the hawksbill worktree, with copies of most on the Frame under `experiments/room-reconstruction/`: `20261106-ir-yaw-scale-field-audit`, `20261106-far-target-ir-yaw`, `20261106-field-mesh-replay`, `20261106-stationary-dot-shift`. Each has a `README.md`.
  - **The fixed −0.086° correction should not be reused.** The motion-to-tag gap follows per-session SLAM/tag-PnP scale (lambda 1.0145, 1.0219, 0.9769), predicted within 0.02° across a sign flip. Its 0.0017° agreement on the retake was coincidence. This corrects the "fixed correction" suggestion in E05.
  - **Far-target unlit yaw worked in development.** Frozen rule passed on one scan: −0.055°, far samples alone −0.056°, depth-subset range 0.006°. Without far samples, tag-free scale and yaw are fragile. Not held-out, thin tag reference, and near range untested.
  - **A fitted image field of about ±1 px helps the motion estimator but not depth.** The frozen replay verdict was "estimator only". Do not apply it to depth or meshes. Its cause is unknown.
  - **Dot matches read nearer than room-light texture on the same scene,** about 0.15–0.25°, or about 60 mm on the wall take at one yaw. A stage-by-stage test places it at unlit-to-lit, not the prior, subtraction or plane fit. Still-headset takes show it without motion (post hoc; the frozen verdict "absent" came from a weak definition of dot-dominated pixels). So unlit methods near −0.13° to −0.17° and the −0.30° dot-plane value on the wall are different quantities. This corrects treating −0.30° as the tag yaw. Unlit yaw plus about 0.22° predicted plane yaw on wall and desk.
  - **Open:** which of dots or texture gives true depth, and why they differ. Most of a far-target or 4 ms comparison carries little dot signal, since 4 ms pulses gave no measurable dots in the still scene, so no fixed offset is validated.
  - **Until resumed:** keep per-session dot-plane yaw for meshes, use 10 ms pulses for dot work, and treat the emitter as unmeasured: the V2 board has no hardware timeout and must not be described as Class 1. Next tests when resumed: a dim-light still take, then disparity against inverse distance at tape-measured positions. The held-out scan waits on those. Each recording needs user approval.
- **Resolution plan, in order:**
  1. A passive logger: a 1 s timed tag capture every few minutes during normal use, with calibration snapshot, thermal zones and uptime. It establishes whether yaw follows temperature, wear or strap tension, or time.
  2. A 5-minute strap test.
  3. A per-unit temperature model, if the correlation holds.
  4. Online motion self-calibration of yaw: metric multi-view consistency with tracked poses. Pitch and roll can be checked continuously from vertical residuals. E10A did this for the **IR** pair against a controller-traced wall (zeros −0.41° that session); the IMX616 pair is still open.
  5. Validate on held-out tag captures across sessions. Gate: |disparity excess| ≤ 0.3 px at 0.7–2.5 m, or the G plane-offset tolerance, whichever is tighter.

## Research assessment and rationale

**The strongest match combines better temporal depth filtering with explicit plane-aware mesh optimization, while keeping the current stereo model.** Meta’s CVPR 2023 point-based fusion paper directly refines existing depth; JPL’s inverse-depth filter offers a lighter geometric approach; Voxfield improves projective TSDF surface estimation. For the user’s added priority of accurate large flat surfaces, PlanarMesh and Wang/Guo’s plane-based RGB-D optimization are more direct references than generic mesh smoothing. The rig is calibrated; the separate open question is whether the model’s uncertainty-like output reliably predicts errors. Current adjacent source already contains optional sparse TSDF and motion-warped history. My recommendation is robust reprojection and fusion first, followed by plane-constrained simplification on supported walls, floors and tabletops. This ranking reflects architectural fit, not a measured improvement on Frame.

## Explicit planes address large flat surfaces directly

**Yes: plane-aware reconstruction can both straighten well-supported planar geometry and replace dense wall triangles with larger polygons.** These are separate benefits. Ordinary triangle decimation reduces mesh size, and smoothing reduces variation, but neither establishes the correct plane. The useful objective is to estimate a plane from reliable multiview observations, constrain only compatible geometry to it, preserve its boundary, and leave nonplanar details intact. A calibrated rig supplies the geometry for this process; systematic depth bias and pose errors still limit accuracy.

**PlanarMesh, IROS 2025, is the most direct online meshing reference for this added requirement.** It extracts planar patches and constructs an incremental mesh from them, making large flat regions a first-class representation. Its official implementation uses C++17, PCL, CGAL and OpenMP and is GPLv3. The published setting is LiDAR, including a 2 Hz 64-beam configuration, so it is an algorithmic starting point rather than a demonstrated neural-stereo solution. Adapting it would require confidence-aware surface samples, careful handling of missing depth, and evaluation of plane support under stereo errors. [Paper](https://arxiv.org/html/2510.13599v1), [official code](https://github.com/ori-drs/planar_mesh).

**Wang and Guo’s Plane-Based Optimization of Geometry and Texture for RGB-D Reconstruction, 3DV 2018, is the closer reference for improving an existing mesh.** It accepts a dense mesh and posed RGB-D observations, uses plane primitives, simplifies the mesh, and jointly optimizes geometry and texture. Its MIT-licensed implementation is particularly useful for an optional scan-finalization pass. The reference is offline and memory-heavy: its documentation reports roughly 20 GB for a one-million-face input, which rules out assuming a direct on-headset port. Geometry-only adaptation is an engineering task; the published pipeline also uses color. [Author paper](https://personal.utdallas.edu/~xguo/3DV2018.pdf), [official code](https://github.com/chaowang15/plane-opt-rgbd).

**AirPlanes, CVPR 2024, is a useful learned plane-estimation comparator.** Its 3D-consistent embeddings help identify planes, but its reference pipeline uses SimpleRecon, a scene MLP and RANSAC rather than serving as a small drop-in TSDF mesher. It is a higher-compute experiment for plane support and segmentation, not the first on-device change. [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Watson_AirPlanes_Accurate_Plane_Estimation_via_3D-Consistent_Embeddings_CVPR_2024_paper.html), [official code](https://github.com/nianticlabs/airplanes).

For Frame, I would first fit persistent plane hypotheses to temporally validated samples, require multiview support, and jointly refine their offsets and normals against reliable observations. Constrain simplification by measured residual and boundary error. Keep doorframes, corners, chair legs and curved objects outside a plane unless evidence supports inclusion. Triangulate planar interiors coarsely while preserving outlines, holes and intersections. Retain TSDF detail elsewhere, without forcing every surface into a Manhattan layout. This hybrid recommendation borrows the representation strategy; it does not promise the papers’ accuracy on this rig. Evaluate plane offset, normal error, boundary displacement and triangle count alongside ordinary mesh error. A flatter wall is useful only if it is in the right place.

## Additional papers for fusion and meshing

These papers extend the earlier plane, temporal-depth and TSDF references. They are candidate methods or evaluation guidance, not measured improvements on Frame. Priority reflects fit to the current posed-stereo and sparse-TSDF pipeline.

| Paper | Relevant idea | Project use and test |
|---|---|---|
| [SurfelMeshing — Schöps et al.](https://arxiv.org/abs/1810.00729) | Fuse depth into surface elements (*surfels*) and asynchronously retriangulate affected areas. Its representation accommodates varying sampling density, thin objects and pose corrections. | **Highest-priority alternate mesher for E12/E18.** Compare with the same calibrated depth, poses and scene set as TSDF/surface nets; measure thin-object survival, openings, mesh age, triangle count and memory. This changes the fusion representation too, so do not attribute any gain solely to triangulation. |
| [PSDF Fusion — Dong et al., ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Wei_Dong_Probabilistic_Signed_Distance_ECCV_2018_paper.php) | Represent signed-distance uncertainty and inlier probability, using a voxel/surfel/mesh hybrid for online reconstruction. | **E07/E12 reference.** Compare calibrated reliability and robust updates against the current capped-weight TSDF before considering the full hybrid structure. Preserve raw observations for a fair replay. |
| [BundleFusion — Dai et al.](https://arxiv.org/abs/1604.01093) | Globally correct camera poses and reintegrate surfaces to keep geometry consistent after loop closure or tracking correction. | **E18/E19 reference.** If capture poses change, rebuild from retained keyframes or perform validated de-integration/reintegration. Test a looped scan for duplicate or thick walls. Its full tracking system is not a drop-in requirement. |
| [Probabilistic Volumetric Fusion — Rosinol et al., WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Rosinol_Probabilistic_Volumetric_Fusion_for_Dense_Monocular_SLAM_WACV_2023_paper.html) | Propagate estimated depth uncertainty into volumetric fusion weights. | **E03/E07 reference.** Test whether calibrated stereo uncertainty improves held-out mesh accuracy and completeness over simple confidence weighting. Its uncertainty comes from monocular SLAM optimization, so its numerical model does not transfer directly. |
| [TSDF++ — Grinvald et al.](https://arxiv.org/abs/2105.07468) | Represent multiple object surfaces at the same volume location to preserve geometry behind moving occluders. | **Conditional E07 reference** if hands/people corrupt already observed walls or tabletops. First test simpler motion rejection and contradiction rules. TSDF++ addresses dynamic-object occlusion; it is not evidence that ordinary thin static surfaces will be fixed. |
| [VDBFusion — Vizzo et al.](https://www.mdpi.com/1424-8220/22/3/1296) | Sparse TSDF integration with OpenVDB, including RGB-D experiments and mesh extraction. | **CPU replay/reference candidate for E07/E08.** Compare map memory, integration cost and output quality under matched voxel/truncation settings. Published CPU rates cannot predict Frame performance. |
| [Tanks and Temples — Knapitsch et al.](https://jaesik.info/publications/data/17_siggraph.pdf) | Evaluate reconstruction with geometric precision, recall and their F-score at a declared distance threshold. | **E01 evaluation guidance.** Add thresholded precision/recall/F-score to local measured scenes; report both terms and the threshold, alongside plane, boundary and topology checks. Its benchmark scenes are not a substitute for headset-specific captures. |

The most informative next comparison is **current TSDF/surface nets versus a SurfelMeshing-style replay**, with identical depth frames and poses. In parallel, PSDF Fusion and probabilistic fusion motivate an uncertainty-weighting ablation within the existing TSDF. If pose corrections are introduced, BundleFusion makes reintegration a separate required experiment. None of these comparisons should replace the runbook's independently measured geometry gate.

## Existing code makes filtering the first intervention

The recovered model produces float disparity and an auxiliary uint8 uncertainty-like head. With the calibrated baseline and rectified focal length, metric conversion follows **z = fB/d**. Disparity is proportional to inverse depth, making inverse-depth filtering especially natural here. The auxiliary head’s role is a different issue: geometric rig calibration does not establish that its values are variances, probabilities, or log uncertainties. The inspected runtime currently converts that head into confidence with a clipped linear heuristic. That is useful as an ordering cue, but it is not sufficient evidence for inverse-variance fusion. [Current depth pipeline](/Users/owenelliott/Documents/projects/frame-spatial-runtime/src/model_depth_pipeline.cpp:130).

The adjacent runtime already implements a sparse projective TSDF with **3 cm voxels, 12 cm truncation, maximum weight 32, and surface-nets extraction**. It consumes confidence and capture-time poses. In the inspected application source, `--tsdf` and `--model-temporal` are both opt-in; source inspection does not establish the flags or build running on the device. The older spatial-stuff README’s description of simple point averaging is therefore insufficient to describe the latest available implementation. [TSDF configuration](/Users/owenelliott/Documents/projects/frame-spatial-runtime/include/tsdf_volume.hpp:25), [application options](/Users/owenelliott/Documents/projects/frame-spatial-runtime/src/openxr_point_cloud.cpp:1087).

There are also two different temporal mechanisms to evaluate. The existing history path warps previous disparity into model input state; an external depth filter would compare current measurements against an independently maintained geometric history. They can complement one another, but shared history correlates predictions. A mesh integrator should not treat every recurrent prediction as fresh independent evidence. The inspected history also box-downsamples disparity before splatting, which can mix foreground and background at boundaries. These observations identify experiments, not a proven diagnosis of the current visual failures. [Temporal implementation](/Users/owenelliott/Documents/projects/frame-spatial-runtime/scripts/model_temporal.py:1).

## Four papers cover the most useful changes

| Priority | Paper and researchers | Why it fits | Main constraint |
|---|---|---|---|
| 1 — closest Meta match | Khan, Penner, Lanman, Xiao, Reality Labs, **Temporally Consistent Online Depth Estimation Using Point-Based Fusion**, CVPR 2023 | Refines existing depth through persistent point history and learned fusion | RGB-dependent pretrained networks; desktop GPU reference |
| 2 — first lightweight design | Cigla, Brockers, Matthies, JPL/Caltech, **Gaussian Mixture Models for Temporal Depth Fusion**, WACV 2017 | Pose-aware inverse-depth hypotheses match stereo disparity well | Full implementation and target timing were not verified |
| 3 — backend upgrade | Pan et al., **Voxfield**, IROS 2022 | Corrects projective TSDF geometry using surface normals | Reliable normals needed; port and benchmark required |
| 4 — learned fusion experiment | Weder et al., **RoutedFusion**, CVPR 2020 | Learns to route noisy depth and update an existing volume | Training/domain transfer and added inference cost |

**Meta’s point-based fusion is the most directly relevant Meta paper.** It retains a global point cloud, renders history into the current camera, detects temporal changes, and learns how to combine current depth with prior geometry. It accepts external depth predictions, so the Model-derived stereo network can stay. The official loader explicitly supports disparity-to-depth conversion using focal length and baseline. This is a depth-refinement frontend, not a room-mesh generator. [Paper](https://arxiv.org/html/2304.07435v1), [official code](https://github.com/facebookresearch/TemporallyConsistentDepth).

Its practical limitations matter. The two fusion networks contain approximately 4.45M and 4.44M parameters, with reported network times of 7.2 ms and 5.1 ms at 512×512 on an RTX 3080. Those figures are not Frame timings. Pretrained fusion uses RGB as well as inverse depth; grayscale replicated across channels would be an unvalidated domain shift. The authors also caution about larger inter-frame motion. I would test it on recorded sequences on a desktop before porting or distilling it. The released code is CC BY-NC 4.0. [Paper](https://arxiv.org/html/2304.07435v1), [repository and license](https://github.com/facebookresearch/TemporallyConsistentDepth).

**JPL’s Gaussian-mixture filter is the best lightweight conceptual starting point.** It represents per-pixel inverse depth with multiple Gaussian hypotheses and transforms observations using camera motion. Multiple hypotheses are valuable when foreground and background compete at a depth discontinuity: averaging them produces a surface that does not exist. Its egocentric image-space representation also avoids requiring another depth network. The NASA record supports this architecture, but the full update rules, available code, and hardware timings were not verified. Treat it as a research foundation for a bounded robust filter, rather than a ready-made library. [NASA paper record](https://ntrs.nasa.gov/citations/20210007922).

**Voxfield is the most targeted classical backend upgrade.** Standard projective TSDF distance is measured along the viewing direction; Voxfield uses a non-projective formulation with normal information to address resulting surface-distance errors. This fits the existing hash-volume design better than adopting another complete SLAM system. Its authors demonstrate onboard robotic mapping, but that is not evidence of Frame performance. The BSD-3 code is a Voxblox extension; the mathematical update can be studied without importing its ROS application stack. [Paper](https://lucabartolomei.github.io/publications/IROS_2022_Voxfield.pdf), [official implementation](https://github.com/v4rl-ucy/voxfield).

**RoutedFusion is the strongest learned depth-fusion comparison.** Its routing network estimates cleaned depth and confidence, while its fusion network predicts volume updates from incoming depth and existing geometry. That directly addresses noisy stereo integration and preserves the current predictor. It would need evaluation on this model’s structured, temporally correlated errors; synthetic noise performance is insufficient. Its reference environment is GPU-oriented, and the repository requests contact for commercial use. NeuralFusion extends this direction with a latent feature volume, but adds representation memory and decoding complexity that make it a lower priority here. [RoutedFusion paper](https://arxiv.org/abs/2001.04388), [code](https://github.com/weders/RoutedFusion), [NeuralFusion paper](https://openaccess.thecvf.com/content/CVPR2021/html/Weder_NeuralFusion_Online_Depth_Fusion_in_Latent_Space_CVPR_2021_paper.html).

## Motion and uncertainty need separate treatment

For moving people or hands, Meta/JHU’s **CODD: Temporally Consistent Online Depth Estimation in Dynamic Scenes**, WACV 2023, is a useful second Meta reference. It combines stereo with learned per-pixel motion and fusion, rather than assuming all change comes from camera motion. Its reported 25 FPS at 640×480 is on a Titan RTX and includes its own stereo pipeline. The motion stage makes it a heavier intervention than a rigid reprojection filter; it also cannot recover a correct surface when both current and prior estimates are wrong. Use it when dynamic ghosting remains the dominant failure after a geometric baseline. [Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Li_Temporally_Consistent_Online_Depth_Estimation_in_Dynamic_Scenes_WACV_2023_paper.pdf), [official code](https://github.com/facebookresearch/CODD).

MIT’s **UfM***, a May 2026 preprint by Sudhakar, Karaman and Sze, is an interesting uncertainty layer. It estimates reliability from disagreement across posed views without retraining the depth predictor. The paper reports 30 FPS at 224×224 for uncertainty on two Cortex-A76 cores; depth inference in that demonstration uses a separate IMX500 accelerator. It is an uncertainty estimator, not a denoiser or mesher. Consistently wrong predictions can remain falsely reassuring, and transfer from its evaluated depth models to recurrent stereo is unproven. A simple reprojected residual score should be the baseline it must beat. [Paper](https://arxiv.org/html/2605.23098v1), [earlier UfM code](https://github.com/mit-lean/ufm).

DynamicStereo and DeepVideoMVS are useful architectural references, but they replace the depth-estimation frontend. DynamicStereo’s published configuration reports 1.2 seconds per 720p frame on a GPU; that makes it more useful as an offline comparator than a direct headset recommendation. DeepVideoMVS demonstrates motion-compensated recurrent fusion, without establishing the recovered Model model’s state contract. Neither is the first change I would make. [DynamicStereo](https://dynamic-stereo.github.io/), [DeepVideoMVS](https://openaccess.thecvf.com/content/CVPR2021/html/Duzceker_DeepVideoMVS_Multi-View_Stereo_on_Video_With_Recurrent_Spatio-Temporal_Fusion_CVPR_2021_paper.html).

## A staged replay experiment can select the implementation

The proposed pipeline is **existing disparity → visibility-aware temporal filtering → validated depth and reliability → robust existing TSDF → incremental mesh extraction**. This is an engineering recommendation synthesized from the papers. Reproject bounded history using capture-time poses, z-buffer competing surfaces, compare inverse-depth residuals, and reject invalid correspondences near occlusion boundaries. Reset stale history after tracking loss or large pose gaps. Keep new foreground observations separate from the background until they are resolved; disagreement should not automatically mean “trust the old map.”

First replay identical captures through the existing TSDF with recurrence disabled and enabled, then add external robust filtering. Empirically fit how the uncertainty-like head relates to observed disparity error on held-out scenes. Until that relation is established, use it conservatively as a reliability cue. If disparity standard deviation becomes available, first-order propagation gives **σz ≈ fB·σd/d²**; that derivation explains range-dependent weighting without assuming the head already encodes σd. Cap history confidence and require meaningful viewpoint change before increasing durable map certainty.

Next strengthen integration with residual-dependent weights and persistent contradiction handling. Moving objects should remain short-lived until supported as static geometry. Repeated free-space evidence can remove old surfaces, while a single bad prediction should not carve holes in walls. Then test Voxfield’s normal-aware update, estimating normals only in valid local neighborhoods. Tune voxel size and truncation against measured prediction error. Smaller voxels and a different polygonizer cannot repair biased measurements; first improve the field being meshed. For runtime, cache dirty blocks rather than rebuilding the entire surface each extraction cycle. [Current integration and extraction scheduling](/Users/owenelliott/Documents/projects/frame-spatial-runtime/src/cloud_worker.cpp:723).

Evaluate stationary walls, headset motion, oblique planes, thin chair legs, textureless surfaces, rapid turns, and people crossing the scene. Measure pose-aligned depth jitter alongside geometric error, boundary thickness, thin-object survival, ghost lifetime, mesh accuracy/completeness, latency, and memory. Report predictor, filter, integration, and meshing costs separately. The decisive result is a more accurate persistent mesh with acceptable responsiveness: reduced flicker alone can conceal lag or stable bias. Use Meta’s learned filter and RoutedFusion as offline challengers to that strengthened geometric baseline before committing to additional on-device networks.

The practical choice is therefore **JPL-inspired inverse-depth filtering first, Meta point-based fusion as the quality comparator, and plane-aware simplification for large flat surfaces**. Voxfield then tests whether the volume update itself remains a significant source of error. This sequence separates depth quality, temporal decisions, surface estimation and mesh complexity so each change has a measurable purpose.

## Implementation plan for a dedicated room scanner

The intended product is a dedicated room-scanning application. No game runs during reconstruction. The user sees a live mesh preview; after scanning, the application refines and saves a persistent mesh for games to load. This changes the scheduling priorities: reconstruction can use a substantial share of the GPU during capture, and expensive optimization can run after capture ends. The compositor, tracking and preview still require bounded responsiveness. Runtime figures below are **unmeasured engineering budgets**, not paper benchmarks or predictions of measured Frame performance.

### Execution modes and persistent output

1. **Capture:** acquire calibrated stereo and capture-aligned poses, run the existing predictor, reject inconsistent depth, fuse observations, and update a progressive preview. Scan status should distinguish observed coverage from unknown regions.
2. **Finalize:** stop acquiring new observations; refine supported planes, crease lines and boundaries; regenerate affected mesh regions; simplify within geometric error limits; validate topology and generate rendering/collision versions.
3. **Save and load:** write a versioned indexed mesh, normals, observed/unknown boundary metadata, calibration/configuration identifiers, units, winding, coordinate convention, and a persistent spatial-anchor reference. A session-relative stage coordinate alone is insufficient for loading the mesh in a later session. Use the platform's persistent localization/anchor mechanism where available and verify relocalization before using saved geometry.

Store a recoverable scan checkpoint before finalization and atomically publish the finalized asset. Keep the preview, saved render mesh, and collision mesh as separate derived representations. Avoid serializing an unbounded history or raw camera DMA buffers into the game-facing asset. Any material geometry change after finalization creates a new asset version.

### Ordered implementation work

| Phase | Concrete implementation | Validation gate | Basis |
|---|---|---|---|
| 1. Replay and timing | Extend recording with model outputs, rectified images, pose/timestamp provenance, sequence, calibration hash and configuration. Compare recurrence off/on independently of external filtering. Extend the bridge protocol for optional second disparity plane. Verify exposure timing rather than assuming buffer-read midpoint equals exposure. | Identical observations reproduce baseline geometry and all ablations. | Existing `model_temporal_replay.py`, `model_depth_pipeline.cpp` and `cloud_worker.cpp`; measurement design is project-specific. |
| 2. Observation contract | Add `DepthObservation`: raw disparity/uncertainty, metric depth, validity, reliability, normals, edge classes, pose, timestamp, generation and measured/filtered/predicted provenance. Retain native-resolution sampling support after expansion. | Missing or predicted values never become independent measured support. | Engineering design; existing bridge and geometry contracts. |
| 3. Edges and normals | Detect luma-edge candidates; confirm depth jumps and normal changes; estimate normals within compatible neighborhoods. Separate appearance-only boundaries, occlusion boundaries, creases and uncertain candidates. | Paint/shadows remain flat; silhouettes do not bridge; normals do not average across corners. | [Extracting Sharp Features from RGB-D Images](https://people.csail.mit.edu/jiex/papers/SharpFeatures/paper.pdf). |
| 4. Temporal depth | Reproject bounded history, resolve visibility, robustly compare inverse depths, preserve competing foreground/background hypotheses, cap correlated evidence and reset on invalid tracking. Fit uncertainty-score reliability on held-out captures. | Jitter improves without increased edge thickness, lag or ghost lifetime. | [JPL Gaussian-mixture temporal fusion](https://ntrs.nasa.gov/citations/20210007922), [Meta point-based fusion](https://arxiv.org/html/2304.07435v1). |
| 5. Robust TSDF | Extend existing volume with residual weighting, normal-based incidence where valid, conservative boundary updates, contradiction evidence, dirty blocks and bounded allocation. Do not fuse plane-predicted depth as measured evidence. Evaluate non-projective distance separately. | Oblique surfaces improve; transient errors cannot erase walls; removed objects eventually clear. | [Voxfield](https://lucabartolomei.github.io/publications/IROS_2022_Voxfield.pdf), [nvblox](https://arxiv.org/abs/2311.00626). |
| 6. Plane map | Region-grow normal-compatible samples, robustly fit bounded patches, associate across views using offset/orientation/overlap, track covariance and footprints including holes. Maintain tentative/confirmed/contested/retired states. | Stable wall identity without merging disconnected patches, openings or nearby objects. | [PlanarMesh](https://arxiv.org/html/2510.13599v1), [plane-based RGB-D optimization](https://arxiv.org/abs/1905.09829). |
| 7. Adjustable pinning | Refine plane normals/offsets against retained real samples; apply finite confidence-weighted constraints to associated mesh vertices. Keep tracking poses fixed initially. Contested constraints weaken or retire. Preserve the measured TSDF and regenerate constrained output when anchors move. | Early plane errors can recover; no old pinned geometry remains baked into the volume. | Plane-based optimization above; uncertainty-dependent lifecycle is proposed engineering. |
| 8. Structural features | Track bounded crease intersections only where supported patches meet; retain separate normal groups; preserve foreground outlines, openings and unknown regions. | Corners sharpen without closing doorways or joining unrelated surfaces. | Sharp-feature paper above; [PlanarMesh](https://arxiv.org/html/2510.13599v1). |
| 9. Mesh refinement | Start with indexed surface-nets output plus local constraints and boundary-protected simplification. Remesh dirty blocks with neighbor halos and deterministic shared vertices. Then compare feature-preserving dual contouring. | No cracks, flipped/degenerate faces or boundary displacement beyond tolerance; lower triangle count at matched accuracy. | [Dual Contouring of Signed Distance Data](https://arxiv.org/abs/2604.00157). Its sampled-SDF results do not establish behavior on our noisy projective TSDF. |
| 10. Scheduling and GPU | Separate capture/fusion, plane optimization and mesh publication; bound work queues and publish complete generations. Port dense stages to Vulkan; use compact CPU summaries. Reject stale asynchronous results after resets. | Responsive preview, no unbounded backlog, no stale asset publication. | [Khronos synchronization guide](https://docs.vulkan.org/guide/latest/synchronization.html), [async-compute sample](https://docs.vulkan.org/samples/latest/samples/performance/async_compute/README.html). |
| 11. Finalize and persist | Run extra plane/crease iterations, remesh changed regions, constrained simplification and collision-mesh generation; validate geometry, checkpoint and atomically save anchored assets. Support cancellation with a usable preview checkpoint. | Reload/relocalize preserves scale and placement; openings and thin objects survive; validation failures retain the previous asset. | Plane-based optimization and PlanarMesh inform geometry; asset/anchor lifecycle is application engineering. |

Suggested new components are `depth_observation.hpp`, `depth_features.hpp/.cpp`, `temporal_depth_filter.hpp/.cpp`, `plane_map.hpp/.cpp`, `surface_constraints.hpp/.cpp`, `constrained_mesher.hpp/.cpp`, a shared `reconstruction_gpu` Vulkan backend, and scan asset serialization/relocalization support. Extend `tsdf_volume.cpp`, `cloud_worker.cpp`, the Model bridge, replay tools and both build systems. Reuse existing occlusion geometry/history primitives where their contracts fit; dedicated scanning should have its own persistent lifecycle rather than inheriting short-lived occlusion expiry rules. The inspected CMake application lists omit `tsdf_volume.cpp` while Makefile includes it; align these before adding modules.

## Compute cost and GPU allocation

The dense stages have abundant pixel/voxel parallelism. GPU TSDF integration and meshing have a strong precedent in [nvblox](https://arxiv.org/abs/2311.00626), but its [implementation is CUDA-based](https://github.com/nvidia-isaac/nvblox). It is an architectural reference, not an Adreno/Vulkan dependency. Published NVIDIA speedups cannot be used to predict Frame timings. The current runtime already has Vulkan stereo dispatch and staging patterns, but its CPU `unordered_map` volume cannot be moved to a shader unchanged.

| Stage | Initial placement | Frequency during capture | Provisional work budget |
|---|---|---|---|
| Edge maps, local normals, consistency | Vulkan compute | Each accepted depth frame | 1–4 ms GPU per depth frame |
| History reprojection, visibility, robust filtering | Vulkan compute | Each accepted depth frame | 1–4 ms GPU per depth frame |
| Active-block discovery, robust TSDF integration | GPU updates; initially CPU allocation | Each accepted depth frame | 2–8 ms GPU per depth frame |
| Upload, compact readback, submission overhead | Batched transfer/control | Each accepted depth frame | 1–4 ms elapsed allowance; measure separately |
| Plane labels and sufficient-statistic reduction | GPU dense pass, CPU summaries | Included in feature pass / periodic | Shared with feature work; profile before assigning a separate budget |
| Plane association and small local constraint solve | CPU initially | 2–5 Hz, bounded active patches | 2–10 ms CPU per optimization update |
| Dirty-region mesh extraction and local projection | GPU | 5–10 Hz preview goal | 2–10 ms GPU per mesh update |
| Topology repair and complex simplification | CPU initially | Limited local work during capture; full work after capture | Do not put an unbounded pass in the live path |
| Final global plane/boundary refinement and mesh generation | CPU/GPU hybrid | Once capture ends | Seconds to tens of seconds is a product allowance to investigate, not a runtime forecast |

These budgets assume roughly 256×320 model outputs, feature processing at model resolution or carefully bounded rectified resolution, 3 cm-class voxels, an active local map, bounded plane/keyframe counts, and dirty-region meshing. They exclude stereo inference, detiling/rectification not explicitly moved into this backend, full-room reconstruction passes and disk writes. Expected order of magnitude is a **5–20 ms incremental depth-processing budget**, plus **2–10 ms on mesh-update iterations**. This is a target envelope to test, not a guarantee or a p95 bound. Whole-map extraction, finer voxels, heavy learned refinement or unconstrained topology optimization can exceed it substantially.

The dedicated scanner can reserve much more GPU time than an in-game service. Start with the capture rate sustained by the current predictor and a 5–10 Hz geometry-preview update, while the headset continues rendering at its required display cadence. At an illustrative 10 depth frames/s, 20 ms of GPU work per depth frame consumes 200 ms of GPU execution per second: approximately 20% if serialized. This arithmetic excludes meshing, preview, compositor, memory stalls and inference GPU use. It does not mean a 20 ms burst fits an 11.1 ms display interval. Slice active-block and mesh work into bounded submissions, show the last complete preview, and measure missed display frames. Async queues express potential overlap; they do not provide free compute or guaranteed concurrency. [Khronos async-compute guidance](https://docs.vulkan.org/samples/latest/samples/performance/async_compute/README.html).

Do not assign a percentage of total runtime to the GPU until measured. Almost all dense arithmetic can be GPU work; sparse association, allocation policy, topology management and asset I/O can remain on the CPU. Initial port order: features/filtering, TSDF updates, then dirty-region extraction. Keep a CPU reference for deterministic geometry comparisons.

### GPU data flow and implementation details

- Upload the model outputs once per depth frame and retain intermediate observations, history and volume data on the GPU. The existing Python/DSP-to-CPU bridge remains a measured transfer boundary initially; direct DSP-to-GPU sharing is a later investigation, not an assumed capability.
- Replace per-voxel CPU containers with pooled contiguous block storage plus block lookup/indirection. Initially batch CPU block allocation; move allocation/hash insertion to the GPU only if profiling justifies it. Do not copy the whole volume each frame.
- Reduce per-plane statistics on the GPU and read back compact summaries/support samples. Upload only changed plane parameters and constraints.
- A forward warp needs a deterministic nearest-surface resolution strategy and controlled attribute ownership. Capability-check atomics and storage formats; use multipass visibility if necessary rather than assuming floating-point atomics.
- Use appropriate precision: FP32 for accumulation and ill-conditioned geometry solves, compact storage where validated. Normal groups and constrained cell solves need robust fallbacks.
- Use counters plus prefix sums/compaction for variable mesh output; enforce buffer capacity and overflow behavior. Dirty-block halos and shared-boundary rules must match the CPU reference.
- Batch dependent compute passes, use explicit barriers, timestamp queries and versioned buffers; avoid a CPU wait/readback between every stage. Rendering must not consume a mesh buffer still being written. [Khronos synchronization](https://docs.vulkan.org/guide/latest/synchronization.html).
- Keep GPU mesh buffers renderable without full CPU readback where compatible with the viewer device. Verify whether the compute backend and preview share the same `VkDevice`; separate devices require deliberate external-memory/synchronization support or an explicit staging fallback. Read back the final asset once at save time.
- Measure sustained thermal behavior, inference throughput, preview cadence and total pipeline age together. A GPU port can reduce CPU time while increasing contention or power; throughput alone is insufficient.

### Memory budget

Memory is likely a larger design constraint than the small plane solves. Derived from the inspected layout: 300,000 leaves × 64 voxels × (2-byte SDF + 2-byte weight + 1-byte luma) is **96,000,000 bytes, about 91.6 MiB**, before leaf metadata, CPU hash nodes, GPU lookup storage, mesh buffers and histories. The header's approximately 20 MB voxel-data comment is inconsistent with the configured maximum. Sparse-map occupancy, not only leaf payload, must be profiled.

A 32-byte observation payload is 2.5 MiB at 256×320, versus about 11.6 MiB at 616×616. Three full-resolution histories would therefore use about 35 MiB before scratch. A 200,000-triangle unindexed preview with 24-byte vertices needs about 13.7 MiB; double-buffering doubles that. These are layout calculations, not measured allocations.

Use **128–384 MiB as an initial reconstruction allocation target**, excluding the existing model/runtime, compositor, camera and application memory. It is a design envelope, not a fit guarantee. Bound active blocks, keyframes, raw support samples, mesh output and scratch; avoid retaining full CPU and GPU maps plus several snapshots simultaneously. Persist checkpoints and free capture-only buffers before large finalization passes. If finer voxels are needed, measure active-band scaling; halving voxel size can multiply volume costs and must not be treated as a cosmetic quality setting.

## Acceptance and rollout

Build synthetic tests with known planes, non-right-angle corners, openings, thin and curved objects, moving occluders, noise/bias, invalid poses and resets. Add real calibrated scans with independent measurements. Measure plane offset/normal error, boundary displacement/thickness, completeness, jitter, ghost lifetime, triangle count at matched error, peak memory, stage p50/p95, mesh age and missed preview/display frames. Missing measurements must not count as free space; fitting a plane must not close an unobserved doorway.

Provisional quality goals from the implementation plan are 50% lower planar jitter and 70% fewer supported planar-interior triangles, conditional on no material regression in plane position, boundaries or completeness. These are project goals to validate against baseline, not literature results. Pinning must recover from incorrect early surfaces; saving/loading must preserve placement through verified persistent localization. Stage the rollout: reproducible baseline → temporal/edge improvements → robust integration → planes and adjustable constraints → incremental constrained meshes → GPU extraction → finalization and saved assets. Learned Meta/RoutedFusion refiners remain optional offline comparisons until the geometric path is measured.

## Optional accelerator refinement after scanning

After the scan ends and all queued stereo inference has drained, the same accelerator can potentially run a second learned model over retained keyframes. This is different from assuming it is free immediately after each camera capture: stereo inference still needs it then. Keep model loading, worker ownership, DSP memory and synchronization sequential until safe multi-model scheduling is demonstrated. The recovered Model execution paths establish execution of that graph, not a general-purpose deployment path for arbitrary new models.

The best architectural fit is a fixed-shape **2D depth/reliability refinement network**, with geometric reprojection and map operations on GPU/CPU. Standard convolutional arithmetic is a plausible accelerator workload; actual suitability requires graph/operator inspection, export, successful compilation or backend implementation, precision validation and measured end-to-end execution. No NPU latency or speedup is established here.

| Candidate | Evidence and fit | Deployment judgment |
|---|---|---|
| Meta point-based fusion, CVPR 2023 | Refines external depth with image-space learned temporal/spatial fusion and persistent geometric history. Its fusion architecture includes residual layers, U-Net structure and layer normalization. [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Khan_Temporally_Consistent_Online_Depth_Estimation_Using_Point-Based_Fusion_CVPR_2023_paper.pdf), [code](https://github.com/facebookresearch/TemporallyConsistentDepth). | Closest first learned-refiner comparison. Offload neural subgraphs only; keep point reprojection/update elsewhere. RGB dependence, normalization, quantization and our stereo noise distribution need validation. |
| RoutedFusion routing stage, CVPR 2020 | Its 2D routing network cleans depth and predicts confidence; the complete method also learns volume updates. [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Weder_RoutedFusion_Learning_Real-Time_Depth_Map_Fusion_CVPR_2020_paper.pdf). | The routing stage is a better initial accelerator target than the whole fusion system. Extracting it is an adaptation and must be evaluated; it does not inherit the full system's reported results. |
| V-FUSE, ICCV 2023 | Accepts multiple posed depth/confidence maps and improves them using support, occlusion and free-space visibility constraints. It uses a learned search-window component and a 3D convolutional fusion architecture. [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Burgdorfer_V-FUSE_Volumetric_Depth_Map_Fusion_with_Long-Range_Constraints_ICCV_2023_paper.pdf), [official code](https://github.com/nburgdorfer/V-FUSE). | Particularly relevant to whole-scan refinement, but volumetric activations, 3D operators and geometric construction make accelerator deployment less straightforward. Start as a desktop quality comparator; do not assume the current backend can run it. |

A proposed smaller alternative is to distill a successful multiview refiner into a bounded 2D network. GPU preprocessing supplies reference inverse depth, reliability, reprojected neighboring depths, visibility masks, disagreement and optional image/normal cues. The accelerator predicts a bounded inverse-depth residual and observation reliability; CPU/GPU validates them and rebuilds the fused map. This is a new model/training project, not an existing drop-in algorithm. Train and evaluate on the current stereo model's structured errors, with scene-disjoint validation. INT8 may be attractive but must preserve metric/boundary accuracy; compare supported higher precision before choosing it.

Post-scan sequence:

1. Stop capture and drain stereo work; checkpoint the preview and release its accelerator graph/resources safely.
2. Select a bounded set of keyframes with coverage and useful baseline diversity, retaining raw observations, calibration and capture poses. Keep images if the chosen refiner requires them. A saved triangle mesh alone is insufficient input for these methods.
3. GPU-reproject supporting views and build validity/visibility features. Schedule fixed-shape refinement sequentially on the accelerator; batch when actual backend support and memory allow it.
4. Check corrected depths against independent views, especially near silhouettes and thin geometry. Limit updates to supported observations and retain correction provenance.
5. Reintegrate accepted refined observations into a fresh volume or affected regions with reconstructable history. Do not simply add refined versions to the old volume and count them as new independent evidence. The existing capped-weight FP16 volume cannot be assumed to support exact subtraction of historical updates.
6. Run plane/crease constraints and final mesh generation on GPU/CPU; validate, simplify and save the anchored asset. Offer cancellation/fallback to the original checkpoint.

NPU work complements rather than replaces mesh optimization. Plane association, robust small solves, variable-topology triangulation, sparse allocation and boundary-constrained simplification remain better first implementations on CPU/GPU. Measure graph load/switch time, geometric preprocessing, transfers, neural execution, reintegration, peak memory, sustained power and total finalize time. For N selected keyframes the serial critical path includes N times their dependent preprocessing/refinement cost plus rebuild and final mesh work; short isolated neural inference does not establish a short whole-room pass.

Adopt the learned stage only if it improves plane position, normals, boundary error and completeness over the geometric baseline. Lower jitter or a visually sharper mesh alone is insufficient. Prefer a refiner requiring minimal backend additions; use V-FUSE or another higher-cost model as a teacher/comparator if it provides a measurable quality benefit that merits distillation.

### Citation scope

Linked papers support the underlying filtering, plane, feature and extraction approaches. The complete combination, module layout, lifecycle, budgets, memory limits and rollout are proposed engineering for this project. Neither the papers nor this document establish measured end-to-end Frame accuracy, latency, power or runtime improvement.
