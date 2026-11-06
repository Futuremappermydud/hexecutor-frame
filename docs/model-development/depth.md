# Depth model and tooling

The depth workspace owns model contracts, extraction/lowering tools, a CPU reference, simulator integration, and future training work. The runtime owns camera geometry and metric interpretation; Hexecutor owns native execution. Raw contexts, weights, recovered graphs, and captures are local artifacts.

## Tensor contract

| Input | Native representation | Meaning in the current adapter |
| --- | --- | --- |
| `x`, `y` | uint8 `[1,1,320,256]` | Left/right model images; raw byte values, with context quantization describing their real-value scale |
| `hidden` | float32 `[1,2,320,256]` | Externally prepared state/history |
| `t` | float32 `[1,2,4,4]` | Model transform/state input |

The input image quantization is approximately `1/255` with zero point 0. Low-level output 0 is uint8 `[1,2,320,256]`; output 1 is float32 of the same shape. An outer wrapper reverses their order. Synthetic stereo probes support interpreting the float head as disparity and the byte head as uncertainty-like; neither is directly a metric-depth map or a calibrated confidence probability.

Keep the transform convention and state initialization explicit. The live native bridge uses zero transforms when history is absent; the standalone CPU adapter's omitted-transform default is identity. Those defaults are different and must be overridden when doing a like-for-like numerical comparison.

## Tooling pipeline

`depth-model/tools/` contains context extraction, graph/weight recovery, quantization checks, simulator execution, synthetic probes, and the memory-level reference. `depth-model/python/depth_model/lowering/resident_program.py` and `resident_image.py` lower the recovered program into audited execution representations; the prepared-program tools produce native tables.

The reference reconstructs compiled/tiled arithmetic and layouts. It is not an untiled training network. Source recovery, compatible QNN context loading, exact compiled execution, camera integration, and model training are separate deliverables.

The original primary graph records 5,870 recovered operations and 5,336 tensor definitions; its scheduled native program has 5,068 commands. Do not interchange recovered node counts, logical neural operators, and compiled execution commands when comparing profiles or support.

## Runtime and replacement boundary

The retained device bridge runs the prepared native paths. CPU reference tools remain useful for differential checks but are not a fallback in the live model pipeline. Model-specific headers, lane tables, contexts, constant pools, and manifest identities must be regenerated/provisioned as a matched package.

For distillation/fine-tuning, put new training definitions and export/evaluation recipes in `depth-model/training/`. Establish preprocessing, eye order, projection, output units, state, and uncertainty semantics as part of the new model contract. Updating weights alone does not establish compatibility with an old compiled schedule.

Read [CPU reference](cpu-reference.md), [training/artifacts](training-and-artifacts.md), and [live integration](../runtime/model-depth.md). This workspace does not distribute legacy model assets or claim independent provenance for a replacement solely because its code has generic names.
