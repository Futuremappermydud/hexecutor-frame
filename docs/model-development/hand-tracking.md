# Hand-tracking workspace

Hand tracking currently consists of local model assets, compatibility research, and a training workspace. There is no completed live Frame hand-tracking pipeline. The depth executor's supported variants do not establish support for arbitrary hand models.

## Components a complete pipeline needs

A hand detector chooses candidate regions; pose estimation consumes the relevant views/features and produces hand structure; temporal tracking associates and stabilizes results. Gesture/surface models can add separate outputs. Each stage needs an explicit camera, crop/projection, handedness, joint-order, coordinate-frame, unit, confidence, and state contract before being integrated into the runtime.

The inspected pose plan contains host tensor operations around two delegated graphs. A native delegate executor alone cannot replace its host batched matrix multiplication, reshaping, slicing, concatenation, normalization, and multi-view aggregation. Those operations need their own implementation or an exported replacement architecture that removes them.

## Reusable executor work and gaps

Convolution, blocked memory, data movement, quantization, and the session infrastructure are potentially reusable. Compatibility remains feature-specific. Matching a variant name is insufficient when a pose graph uses different shapes, channel packing, padding, quantization, or argument layouts.

The recorded direct-pose comparison found 22 unmatched operation types and 253 instances for one inspected plan, in addition to outer host operations. Missing families included layout/space-to-depth variants and pooling. Detection and surface/gesture families had different operator-package and normalization/lookup requirements. These are properties of the inspected artifacts, not a general support census for future hand models.

The present executor schema is depth-oriented: seven pools, fixed four-input/two-output integration, a finite operation registry, and model-specific native schedules. Extending the registry alone does not solve multi-delegate orchestration, host operations, or the hand-tracking application contract.

## Training and integration

Use `hand-tracking/training/` for new model definitions, recipes, export tools, and evaluation code. Keep weights under the ignored local artifact locations. Record teacher/base-checkpoint and dataset provenance. Evaluate occlusion recovery, handedness consistency, metric/coordinate accuracy, temporal behavior, and device latency together.

Prefer a deliberate deployment contract for the replacement rather than reproducing every legacy container boundary. When a new operation is necessary, add explicit coverage/budget checks to Hexecutor and validate its alias and scratch behavior before enabling a full graph. Runtime integration must also provide the correct camera timing and presentation transform.

Shared references: [executor architecture](../hexecutor/README.md), [kernels](../hexecutor/kernels.md), and [training/artifacts](training-and-artifacts.md).
