# Composite stereo protocol

The offline composite comparison freezes its synthetic scene families, development/evaluation/regression seeds, real-frame identities, refinement arms, baseline controls, and source/input hashes. The implementation constants live in `runtime/scripts/stereo_composite_common.py`; this page describes the contract rather than duplicating every case ID.

Tune on the development split and evaluate once on the held-out split. Compare the same image pairs under integer and subpixel/refinement conditions, and retain both accuracy/validity metrics and execution cost. Ambiguous/repetitive patterns, independent noise, exposure changes, vertical misalignment, and thin silhouettes are necessary failure cases. A single attractive disparity image is not the acceptance criterion.

The runner snapshots sources, records planned jobs, and verifies completed output inventories/hashes. Analysis must reject missing or changed evidence. New docs/source hashes identify a new run; old bundles remain interpretable from their saved snapshots.

This protocol is an experimental comparison, not a declaration that the composite matcher is the default live backend.
