# Development and validation

Start from the component contract being changed. Source ownership is described in the architecture/runtime/executor guides; the root and runtime `AGENTS.md` files retain instructions for automated contributors. Existing working edits and local model data are not disposable build scratch.

## Match evidence to the change

| Change | Evidence that matters |
| --- | --- |
| Camera/calibration/timing | Sensor identity, exposure/pose association, projection residuals, and real metric references |
| Depth/fusion estimator | Bias, uncertainty/validity, edge leakage, temporal stability, and scene-disjoint geometry |
| Native arithmetic | Scalar/reference agreement across value/shape boundaries and full-model oracle identity where supported |
| Memory/scheduling | Ownership, alias/write footprints, scratch/stack budgets, adversarial ordering, and device cleanup |
| Performance | Alternating paired runs, matched inputs/binaries, cold versus warm timing, and live clock state |
| Presentation/occlusion | Frame freshness, transforms, per-eye depth/submission, and render-loop cadence |

Host checks, simulator checks, and device measurements answer different questions. The host may use a different kernel implementation; the simulator may not reproduce timing or firmware/session behavior. A passing output comparison is not proof that clocks, callbacks, or firmware memory were restored.

## Practical local checks

From the monorepo root, the repository policy checks need no model weights:

```sh
python3 tools/test_model_artifact_policy.py
python3 tools/model_artifact_policy.py --history
```

From `runtime/`, `tools/test_model_depth_bridge.py` exercises native-backend selection and socket framing without DSP hardware. Geometry/temporal source checks can be compiled with the appropriate host toolchain. `tools/test_room_dataset.py` exercises conversion, pose composition, replay, and manifests using an offline fixture. Use the Makefile/source dependencies for the change rather than treating this short list as the whole test suite.

Full model execution and many kernel probes require private generated fixtures or the compatible SDK/device. Document unavailable prerequisites instead of substituting an unrelated passing check. The last cleanup observed five transport lifecycle fixture errors caused by a missing `worker_tids` field; they also occurred in the untouched original checkout. Do not claim that the entire transport suite passed.

## Documentation and reproducibility

Maintain the canonical guides under `docs/`. Component README links and `runtime/docs` are compatibility paths to that content. Three stable stereo protocol filenames remain because experiment manifests hash them; modifying those pages changes the source identity for new runs. Saved experiment bundles retain their own source snapshot and must not be relabeled as new evidence.

Document why an optimization works, its activation/coverage conditions, and what invalidates the result. Keep individual mutation counts and command-by-command probe histories in local reports. Repeated findings such as the AXI domain, L2 mode transitions, active-block footprint choice, and warm plan identity belong in the maintained guides.

The temporal-fusion report, its research-notes directory, and both preserved fusion handoffs are outside this rewrite. Keep their contents and paths unchanged while the author is working on them.
