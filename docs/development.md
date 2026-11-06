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

Use `make check` for the portable C++ host and worker protocol, geometry/temporal checks, documentation links, and artifact policy. The protocol tests exercise v3/v4 replies, temporal inputs, and session reuse/close through an injected inference implementation.

`make check-model` additionally requires private local assets. It validates the program/image readers and compares both models' native host outputs with saved oracle hashes. Offline Python diagnostics remain under the model and executor development packages; they are not part of the application's live path.

A Linux compile check covers the real device-transport branch; the macOS build covers portable APIs and protocol behavior. Neither proves headset inference or firmware restoration. Device validation must include repeated inference, idle close/reopen, worker exit, VTCM and clock restoration, and eye-client recovery with the provisioned pinned artifacts.

## Documentation and reproducibility

Maintain the canonical guides under `docs/`. Component README links and `runtime/docs` are compatibility paths to that content. Three stable stereo protocol filenames remain because experiment manifests hash them; modifying those pages changes the source identity for new runs. Saved experiment bundles retain their own source snapshot and must not be relabeled as new evidence.

Document why an optimization works, its activation/coverage conditions, and what invalidates the result. Keep individual mutation counts and command-by-command probe histories in local reports. Repeated findings such as the AXI domain, L2 mode transitions, active-block footprint choice, and warm plan identity belong in the maintained guides.

The temporal-fusion report, its research-notes directory, and both preserved fusion handoffs are outside this rewrite. Keep their contents and paths unchanged while the author is working on them.
