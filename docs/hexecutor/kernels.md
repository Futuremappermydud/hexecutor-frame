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

## One HMX owner per transaction

HMX accumulator state is shared by hardware threads 0, 1 and 2. If one thread clears and loads the accumulator, a store issued afterwards by any thread returns that thread's product, and the store consumes the state. Interleaving two transactions corrupts both, even when no two instructions overlap in time. Every HMX clear/load/store sequence must therefore run to completion on one thread before another thread starts one. The executor already follows this: HMX stays on the main list, and the worker lanes and helpers do HVX and scalar work only ([scheduling](scheduling.md)). Keep it that way when moving work between lanes. A lock around a whole transaction would be correct, but it gives no extra HMX throughput.

Evidence: FrameNPU `docs/HMX_ACCUMULATOR.md` (2026-11-14). That test used exact integer-valued FP16 tiles on every pair of workers 0/1/2 and on all three at once. Serialized controls passed 8/8. Deferred stores failed 24/24 and concurrent transactions failed 24/24.

## Unavailable HVX IEEE float

The optional HVX IEEE float extension (`-mhvx-ieee-fp`: `.sf`/`.hf` multiply, add and accumulate) is unusable on the Frame's CDSP. The compiler emits these instructions and they execute without trapping, but they write zero to their destination. FP32/FP16 echo instructions and qfloat arithmetic in the same job behave correctly. A Qualcomm QEMU patch series describes this destination zeroing as the behaviour when the extension is disabled. Use qf16/qf32 arithmetic with explicit conversion. Do not build engines with `-mhvx-ieee-fp`.

Evidence: FrameNPU `docs/NATIVE_ARITHMETIC.md` (2026-11-16) on CDSP firmware `4f9fd567…`, the same image as `/opt/steamvr/drivers/cv/bin/linuxarm64/cdsp_sm8650.mbn` here.

## Adapter coverage

Kernel-specific probes establish layout, shape and numerical constraints for each adapter. [Adapter acceptance and source locations](kernel-adapters.md)

Source: [reference handlers](../../hexecutor/native/program/prepared_reference.c), [HMX paths](../../hexecutor/native/program/prepared_hmx_conv.inc), [data movement](../../hexecutor/native/program/prepared_datamove.inc), and [blocked glue](../../hexecutor/native/kernels/blocked/).
