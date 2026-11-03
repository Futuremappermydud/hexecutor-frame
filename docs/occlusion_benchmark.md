# Occlusion benchmark

For the code investigation, additional support-loss replay, and the resolution
and validation plan for all reported issues, see
[occlusion_investigation.md](occlusion_investigation.md).

Run from the Mac workspace:

```sh
python3 scripts/benchmark_occlusion.py
```

The script restarts the occlusion demo with the usual legacy filter, records a
bounded sequence, and replays both temporal filters on exactly the same input.
Add `--temporal confidence` to view the experimental filter live. Wear the headset in an ordinary room with the demo panels visible.
Press Enter when ready. The Mac announces the phases (keep its speakers audible):

| Phase | Duration | Action |
| --- | --- | --- |
| Warmup | 3 seconds | Look at the panels. |
| Still | 8 seconds | Hold head and hands still. |
| Head turn | 8 seconds | Slowly turn about 20 degrees left/right; keep hands still. |
| Hand motion | 8 seconds | Hold head still; pass a hand in front of a panel and withdraw it. |
| Recovery | 3 seconds | Stay still and watch whether stale occlusion clears. |

Record ordinary walls, furniture, and hands, without a phone noise pattern.
Repeat in the same lighting and panel arrangement. Note visual failures alongside
each report: where a real surface failed to occlude, where virtual pixels vanished
incorrectly, alignment while turning, and hand trails. Script phase cues are
polled every half second; the recording's timestamps determine phase boundaries.

Reports are saved in `artifacts/occlusion_benchmark/<timestamp>/`: `summary.txt`,
`frames.csv`, `runtime.log`, and `manifest.json` (settings and binary/shader hashes). Full recordings remain on the headset under
`captures/occlusion/<timestamp>/recording.bin`. Add `--download-recording` to copy
the binary too. Allow up to 2.2 GB per recording and remove old captures as needed.
A recording ends after 30 seconds or 512 frames. Hitting the frame limit early,
queue overflow, truncation, or missing frame IDs makes it unsuitable for a filter
decision; the analyzer marks it incomplete/lossy and the script returns an error.

## What the report measures

- Measured depth and completed input coverage, averaged within each phase.
- Camera-view area supported by the actual 2x2-cell mesh, for each filter.
- Retained depth that disagrees with reliable current stereo measurements,
  normalized by the number of reliable measurements. This is a ghosting proxy.
- Pose-reprojected valid-depth coverage changes as a percentage of the union.
  This includes genuine disocclusion and projection rounding; it is most useful
  in the still segment. It measures depth support, rather than eye-mask flicker.
- p50/p95 capture, rectification, stereo, cleanup, completion, provenance,
  temporal history, mesh, and publication-age timings; recorded depth throughput.

Compare consistency and coverage together: deleting all geometry would improve
contradiction counts while making occlusion useless. The confidence filter should
reduce contradiction persistence during hand motion without a large loss in
supported mesh area. Use the still segment to check whether that costs stability.
If mesh generation/completion dominate p95 time, optimize those stages before
adding a heavier depth model.

There is no ground-truth depth or captured display mask in this benchmark. It
cannot establish absolute depth accuracy, camera-to-eye calibration accuracy,
end-to-end display latency, or true hand response time. Publication age is measured
from the DMA-read midpoint, not sensor exposure. Stage timings include recorder
allocation/copy and disk contention; compare recording runs with each other, and
use regular runtime logs to check performance without recording.

## Data and controls

`FROCCB02` stores explicitly ordered native little-endian uint32/uint64/float32
fields (supported Mac and Frame hosts), without structure padding. Header:
width/height, rectified intrinsics, optical-to-head transform, and phase durations.
Each frame contains capture/read/publication timestamps, pose timestamp, standing
head pose, relative optical transform, flags, stage timings, left/right rectified
luma, cleaned stereo depth, completed pre-history depth, and evidence classes.
The footer contains written/dropped frame counts and completion status. These are
processed image/depth arrays; raw compositor DMA slots are never recorded.

Disk writes run on a separate thread with a two-frame queue. A full queue drops
recorded frames without blocking capture; replay resets history at gaps because
its relative transforms refer to the immediately preceding live frame.
Capture-pose fallback frames use single-frame depth, matching live behavior.

`FRAME_OCCLUSION_TEMPORAL=confidence` enables the experiment; the default remains
`legacy`. Confidence classes are spatial support/provenance heuristics, not
calibrated probabilities: nearby agreeing measured anchors can confirm after two
observations; other measured depth takes three; inferred depth takes four.
Measured contradiction clears an old occluder immediately. Missing inferred
stable surfaces expire after 150 ms, measured surfaces after 200 ms. Pending
observations have a 200 ms expiry so they can confirm at the current depth rate. These choices
can reveal holes when a measurement is wrong, so benchmark before making the
experiment the default.

The script leaves the chosen demo filter running after recording stops. The
default command uses legacy; `--temporal confidence` selects the experiment.
`FRAME_OCCLUSION_BENCHMARK=/absolute/path.bin` enables recording manually;
`FRAME_OCCLUSION_BENCHMARK_SMOKE=1` uses three seconds solely for engineering checks.
For a quick smoke check of deployment, capture, codec, and replay:

```sh
python3 scripts/benchmark_occlusion.py --smoke
```

Replay an existing recording on Frame:

```sh
OMP_NUM_THREADS=8 ./bin/analyze_occlusion_benchmark recording.bin frames.csv
```

The analyzer and live worker accept a reproducible confidence configuration.
Values are strong/measured/inferred observation counts, followed by inferred,
measured, and pending expiry in milliseconds:

```sh
OMP_NUM_THREADS=8 ./bin/analyze_occlusion_benchmark recording.bin frames.csv \
  --confidence 2 2 3 150 200 200
```

The confidence history constructor validates counts in `[1,255]` and expiry in
`(0,1000]` ms. The live worker uses the comma-separated
`FRAME_OCCLUSION_CONFIDENCE=2,2,3,150,200,200` environment variable; invalid live
configuration prints a diagnostic and falls back to the baseline. Confidence
configuration does not change the legacy filter. A capture/tracking gap above
200 ms always resets history, independently of expiry settings.

Exit status 0 means complete, gap-free data; 3 means incomplete/lossy data;
1 means invalid data/I/O error. `make bin/test_occlusion_benchmark` builds the
confidence transition and codec checks.

## Completed recording and temporal sweep (2026-10-07)

The complete recording `20261007-182413` contains 278 frames at 9.26 Hz, with
zero drops, sequence gaps, or pose fallbacks. Six confidence variants replayed
exactly the same cleaned/completed depth, evidence, and poses; all exited 0.
Reports and per-frame CSVs are in
`artifacts/occlusion_benchmark/20261007-182413/temporal_sweep/`.
Stage timings in these reports belong to the original recording; they do not
measure the variants' execution time or predict performance after optimization.

| Variant | Strong / measured / inferred observations | Inferred / measured / pending expiry (ms) |
| --- | --- | --- |
| baseline | 2 / 3 / 4 | 150 / 200 / 200 |
| balanced | 2 / 2 / 3 | 150 / 200 / 200 |
| two-confirm | 2 / 2 / 2 | 150 / 200 / 200 |
| strong-first | 1 / 2 / 3 | 150 / 200 / 200 |
| longer-hold | 2 / 3 / 4 | 200 / 250 / 250 |
| balanced-hold | 2 / 2 / 3 | 200 / 250 / 250 |

Each table entry below is **mesh coverage / contradiction proxy / coverage
change**, all in percent. Contradiction uses reliable measured pixels as its
denominator; coverage change uses the pose-reprojected depth-support union.
Coverage change is not a display-mask flicker measurement.

| Variant | Warmup | Still | Head turn | Hand motion | Recovery |
| --- | --- | --- | --- | --- | --- |
| legacy | 11.36 / 1.88 / 30.13 | 13.45 / 2.12 / 20.00 | 12.47 / 2.93 / 25.27 | 10.28 / 3.88 / 32.10 | 8.95 / 1.71 / 30.27 |
| baseline | 8.65 / 0.09 / 34.91 | 11.12 / 0.11 / 18.48 | 9.59 / 0.11 / 26.61 | 7.23 / 0.09 / 33.04 | 6.60 / 0.10 / 32.94 |
| balanced | 11.47 / 0.14 / 35.74 | 13.25 / 0.14 / 22.91 | 12.20 / 0.15 / 29.17 | 9.98 / 0.14 / 36.83 | 8.74 / 0.13 / 34.62 |
| two-confirm | 15.84 / 0.17 / 34.65 | 16.79 / 0.16 / 26.87 | 16.25 / 0.18 / 31.14 | 14.28 / 0.16 / 37.21 | 12.41 / 0.15 / 36.56 |
| strong-first | 11.96 / 0.17 / 30.80 | 13.70 / 0.16 / 22.44 | 12.78 / 0.18 / 26.78 | 10.67 / 0.16 / 33.24 | 9.06 / 0.15 / 32.51 |
| longer-hold | 9.58 / 0.13 / 31.19 | 12.45 / 0.15 / 17.76 | 10.81 / 0.15 / 23.41 | 8.34 / 0.12 / 28.80 | 7.54 / 0.12 / 29.37 |
| balanced-hold | 12.53 / 0.18 / 32.28 | 14.63 / 0.18 / 21.07 | 13.53 / 0.20 / 25.75 | 11.03 / 0.17 / 32.06 | 9.70 / 0.16 / 31.25 |

`balanced-hold` (`2 2 3 200 250 250`) is the provisional live-validation candidate.
It recovers mesh coverage above legacy in every phase: still 14.63% versus
13.45%, and hand motion 11.03% versus 10.28%. The hand contradiction proxy falls
from 3.88% to 0.17%, while coverage change is nearly equal (32.06% versus 32.10%).
Still coverage change is slightly worse than legacy (21.07% versus 20.00%) and
worse than baseline confidence (18.48%). This trades the baseline's strongest
still stability for more supported mesh area; it is not a universal improvement.

The baseline remains available by omitting configuration. `two-confirm` has
substantially more mesh but worse coverage change in every phase. `strong-first`
allows strongly supported depth to appear after just one observation, weakening
transient rejection. `longer-hold` alone gives better still stability but keeps
less hand-motion mesh than legacy. No variant changes immediate measured
contradiction clearing, and none makes expiry refresh merely from reprojection.

Longer expiry can retain an unsupported surface an additional 50 ms; the
contradiction proxy only sees reliable current measurements and therefore cannot
prove such holds are visually acceptable. The fixed 200 ms capture-gap reset
still applies. Repeat after the worker/completion changes because update rate
changes the effective delay of observation-count confirmation.

To evaluate the candidate live and record a new full benchmark:

```sh
python3 scripts/benchmark_occlusion.py --temporal confidence \
  --confidence 2 2 3 200 250 250
```

This command selects the same live and replay configuration and records it in
the manifest. Run the full still, head-turn, hand-motion, and recovery sequence,
then record visible holes, false occlusion, head-turn alignment, and hand trails.
Compare with the original baseline and a legacy live run under the same lighting
and panel arrangement. The sweep initially had no headset visual validation;
the subsequent candidate evaluation is recorded below. The runtime's default
remains legacy.

## Candidate headset evaluation (2026-10-07)

The subsequent full candidate run `20261007-183447` used confidence history with
`2 2 3 200 250 250`, completion enabled, luma-only capture, and eight OpenMP
threads. It contains 294 frames at 9.79 Hz, with zero drops, sequence gaps, or
pose fallbacks. The same-input replay reports:

| Phase | Legacy mesh / contradiction / coverage change (%) | Candidate mesh / contradiction / coverage change (%) |
| --- | --- | --- |
| Still | 20.26 / 1.47 / 14.81 | 21.31 / 0.16 / 15.15 |
| Head turn | 18.61 / 2.54 / 16.71 | 19.38 / 0.14 / 17.41 |
| Hand motion | 16.55 / 1.97 / 14.68 | 16.94 / 0.15 / 15.80 |
| Recovery | 15.45 / 4.01 / 17.32 | 15.40 / 0.17 / 19.47 |

Wearer feedback: overall occlusion looks substantially better; some table edges
visibly align with the occlusion. Depth still struggles beyond a few feet.
This is encouraging qualitative evidence, not a measured calibration result.
Additional wearer feedback: long streaks in the point cloud have missing
occlusion for a time, including on nearby surfaces. Streak orientation,
relationship to scene edges, and behavior with a stationary head are not yet
established. This suggests investigating spatial support loss as well as range;
it does not establish whether the gaps originate in capture, matching,
completion, temporal confirmation, or meshing. No specific verdict on hand
trails or head-turn alignment was supplied, and no matched legacy headset
comparison was reported.

### Next investigation: distance-dependent depth loss

The active stereo filter and occlusion surface validation accept 0.25–5 m.
There is no explicit cutoff at a few feet. The current conversion is
`Z = 24.282608 / disparity`: a surface at 1 m has about 24.3 pixels of disparity,
at 3 m about 8.1 pixels, and at 5 m about 4.9 pixels. Smaller disparity makes
depth more sensitive to matching error. The matcher uses local census with
sparse cross aggregation, rather than full SGM. Weak/ambiguous matches,
left-right disagreement, and insufficient neighboring support are rejected.
Mesh cells additionally require all nine samples and a depth spread within
`max(4 cm, 4% of the nearest depth)`.

The next engineering step is to replay the recorded rectified luma through each
stereo/cleanup/completion/mesh stage and inspect where distant surfaces lose
support. Prioritize locating the nearby streaks in these stage outputs too:
mesh cells reject the entire cell when any of nine samples is invalid or its
depth spread fails validation, so narrow input gaps can remove a wider band of
occlusion. Completion also intentionally leaves unknown regions across luma
boundaries, near contradictory anchors, and where overlapping plane predictions
disagree. Inspect both rectified luma images for corresponding bands before
attributing a regular pattern to these later stages. Use marked scene regions
and independently measured distances for a
controlled follow-up; binning only accepted depth cannot measure how much of a
missing distant surface was rejected. Compare textured distant furniture with
plain walls to separate range sensitivity from lack of texture. These are
candidate explanations, not established causes of this visual failure.

Prioritize improving reliable distant stereo support before increasing the
maximum range or extending history expiry. Compare matcher/filter experiments
on identical inputs, preserve foreground boundary rejection, and check both
depth error and visible false occlusion alongside coverage. Recheck timing and
hand response after any change. Reports are in
`artifacts/occlusion_benchmark/20261007-183447/`.

## Configurable lossless recording budget

The recorder retains its two-frame asynchronous queue and v2 codec. The default
budget remains 512 frames. Use `--frame-budget 1024` with the guided script for
roughly 34 Hz over the unchanged 30-second protocol; select a larger declared
budget if measured throughput requires it. The runtime equivalent is
`FRAME_OCCLUSION_FRAME_BUDGET`, an integer in `[1,65536]`. Budget exhaustion
still produces an incomplete footer and fails evaluation. It never shortens the
phases or labels a partial recording complete. Current readers continue to read
original v2 files and permit extended recordings up to the declared hard limit.

Before stopping or starting the demo, the guided script checks disk space for
the entire worst-case uncompressed frame budget plus a 256 MiB reserve. The
manifest records the budget, estimated recording bytes, initial available disk
space, and hashes of relevant sources, binaries, shaders, and calibration.
This check is a preflight estimate; disk failure or competing writes remain
explicit recording failures. Compression/chunking and wearer measurements
remain separate work. `test_occlusion_budget` verifies extended v2 reading,
invalid budget rejection, and incomplete-footer behavior on early exhaustion.


## Guarded recordings (2026-10-07)

`benchmark_occlusion.py --matcher guarded` explicitly records guarded legacy
retention (2500 ms); `--matcher baseline` retains the original comparison mode.
Guarded mode rejects confidence-policy options. The manifest records matcher,
retention, guarded tuning, shader/calibration hashes and generation-log location.
New recordings tag guarded frames in the v2 flags; the analyzer selects the
matching legacy age and labels confidence replay as counterfactual. Old untagged
guarded recordings require external provenance and are not live-parity evidence.
The baseline stage replayer rejects guarded recordings. Use
[guarded matcher replay](stereo_guarded_optimization_results.md) for saved-pair
performance/parity, and the recording analyzer for recorded guarded history.
