# Kernels, layouts, and numerical semantics

Hexecutor combines scalar control, HVX vector operations, and HMX matrix arithmetic. Matrix MAC throughput is only one stage: input layout, weights, quantization parameters, output conversion, and synchronization can dominate a small tiled command.

## Layout-aware execution

Blocked tensors use eight-row tiles and 32-channel groups. u8 and u16 blocked layouts differ; direct flat tensors use a separate address calculation. A tensor view includes maximum dimensions, padding, block span, and slice origin. Equal-looking logical shapes do not establish equal storage or alias behavior.

`prepared_datamove.inc` handles layout transforms, copies, resize, and im2col. Specialized HVX shuffles and row/block traversal replace expensive elementwise address reconstruction where alignment and geometry permit. Overlap checks must describe actual reads/writes. A fast copy that ignores a view's padding or crosses a live alias is a correctness regression even if its isolated output looks plausible.

Convolution paths specialize on input/output type, stride, kernel geometry, packed weight class, and output placement. Dynamic-weight preparation includes scale reduction, inverse scales, packing, and Fi construction. Cache only values whose producer/lifetime proves immutability; a fixed address is not a constant.

## Exact u8 conversion

An early analysis concluded that native-rate HMX byte conversion could not reproduce the host reference's rounding. Later selector/bias-bank work disproved that conclusion. The accurate path extracts two byte planes from the same accumulator: a low wrapped plane and a biased/clipped high plane. HVX reconstructs the needed intermediate and applies the reference requantization.

Relevant selector fields are bit 0 (keep accumulator), bit 1 (wrap rather than clip), and bits 12–13 (bias bank). For the supported range, the two conversions recover `q = floor((acc + bias) / 2^pre)`, allowing the reference result `clip((q*M + 8192) >> 14, 0, 255)`. The recorded path requires `pre <= 7`; unsupported cases use the applicable wide native implementation, not the removed Python backend.

The cheaper `PLANE_SAT` form reduces merge arithmetic while preserving the intended clipping. Output tile alignment and plan-replay Fi construction must use the same rule. A warm-only device failure came from one replay table builder retaining an old bias while the first-run builder had been updated. That is why cold and replay execution both matter.

## u16 and reference identity

HMX halfword conversion, packed weight traversal, scaling, and signedness have their own contracts. Do not generalize a u8 midpoint rule to u16 or assume one native hardware oracle is the simulator's oracle. The source retains a CPU reference and explicit native/host checks for this reason.

Later standalone graph comparisons found Conv, Add, Convert, and a tested dynamic-weight MatMul byte-identical to the compatible simulator on their exercised inputs. The cause of the legacy full-model differences on textured inputs remains unresolved. The old explanation that all such differences are inevitably caused by HMX midpoint rounding should not be revived.

## Adapter coverage

`native_adapter`, `native_kernels`, `native_stride2`, `native_runtime/stride2`, and `native_u16` contain probes and adapters for particular provider layouts. Their results are shape-specific. See [kernel adapters](kernel-adapters.md) before expanding a resolver's accepted feature set.

Source: [reference handlers](../../hexecutor/frame/native_program/prepared_reference.c), [HMX paths](../../hexecutor/frame/native_program/prepared_hmx_conv.inc), [data movement](../../hexecutor/frame/native_program/prepared_datamove.inc), and [blocked glue](../../hexecutor/frame/blocked_glue/).
