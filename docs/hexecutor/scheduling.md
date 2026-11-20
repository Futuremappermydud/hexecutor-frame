# Scheduling and established optimizations

The optimized primary executor keeps a main HMX list, splits independent worker work over two lanes, and serves data-parallel parts through cooperative helpers. It improves an existing compiled schedule; it does not yet produce an unconstrained new allocation plan.

## Dependencies must match memory

Generated schedules include tensor reads/writes, direct ranges, blocked storage, and addresses encoded in command vectors. Spill, Fill, and Zap have meaningful memory effects even when they declare no tensor operands. DMA completion checkpoints also create landing-to-consumer edges.

Whole block tables overstate each command's footprint, because they describe maximum tensor geometry. Exact active-block footprints removed many false dependencies without moving any tensor memory. Host poisoning/diff checks and adversarial serial orders supported the block-level model; row-level write footprints were too narrow for 38 commands. Use the active-block policy, not the tempting narrower row policy.

The measured-cost schedule model moved from 10.37 to 8.58 ms when footprints were tightened. Device time moved from about 10.95 to 9.68 ms. The agreement supports that optimization, while the remaining gap shows that the model does not include all contention and overhead.

## Reuse setup, not mutable values

Prepared metadata avoids repeated descriptor decoding. Expanded block lists are built once; bound hot metadata supports direct command/tensor access. Preflight reuse and no-effect stripping avoid repeated work only while the program identity, bindings, and validated assumptions remain valid.

Per-command plan replay retains geometry-dependent HMX setup across inferences. Memoized weight-copy/layout decisions similarly avoid revalidation, but dynamic weights and scales still follow the current producer chain. Cold initialization and warm replay are separate code paths and must have identical numerical semantics.

Other established steps in the measured recipe include compact staging windows, direct 1×1 reads, supported in-place/sliding paths, reduced pool copies, a single VTCM mapping where selected, selective cache cleaning, and vectorized elementwise/layout kernels. Each depends on its shape and ownership checks. The build recipe records the actual combination.

## Workers and waiting

The normal callback wakes physical threads 0, 1, and 2. The four-worker path uses physical thread 5 as logical worker 3. Threads 3 and 4 are not spare model workers. Thread 4 runs the firmware's CPU request loop. Thread 3 runs the stock eye-tracking loop. Our wake does not stop that loop, and it uses VTCM ([memory layout](memory-layout.md#stock-eye-tracking-thread-and-vtcm)). Only thread 0, 1 or 2 code from the main list may issue HMX ([one HMX owner](kernels.md#one-hmx-owner-per-transaction)). Shared work records use generation/completion counters and bounded joins.

Measured optimization showed that idle/waiting threads could slow busy ones. Longer bounded pauses, fewer unnecessary barriers, and progress publication only where consumers wait improved the total even when individual kernels changed little. Pause 255 is part of the recorded best recipe. This does not justify removing arbitrary ordering: publish/acquire semantics and rare-race risk still require device stress evidence.

## DMA and workspace choices

Tiny weight copies can cost less on CPU than DMA setup, so the recipe has size/variant-dependent decisions. The important `4/30` copy remained faster on DMA: moving it to CPU cost about 0.38 ms in paired runs. A generic claim that DMA or CPU copies are always better is unsupported.

Splitting each u8 convolution into a main staging/MAC job and an independently scheduled merge stayed numerically exact but regressed the device from roughly 9.65 to 9.85–10.08 ms. The schedule model predicted a gain and was wrong for that new handoff structure. Only two merge-ring slots fit the fixed workspace; undersized slots forced slower paths. The rejected split remains a patch, not the baseline.

Likewise, more helper parts, alternate pauses, sparse L2 saves, and moving all non-convolution work off main did not reliably improve the latest recipe. Preserve the reason for a rejected direction instead of repeatedly treating it as untried.

Source: [lane generator](../../samples/depth-model/deployment/lanes/gen_qnn_lanes.py), [memory analysis](../../samples/depth-model/deployment/lanes/mem_layout.py), [best recipe](../../samples/depth-model/deployment/lanes/build_best.sh), and [executor](../../hexecutor/native/program/prepared_executor.c).
