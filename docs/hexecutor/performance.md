# Performance and evidence

There are multiple execution generations in this repository. State the engine, generated schedule, arithmetic oracle, input set, and timing boundary before comparing numbers.

## Distinct baselines

| Path or build | Recorded warm time | Meaning |
| --- | ---: | --- |
| Pinned runtime `prepared-qnn` package | about 23 ms | Primary run list selected by the ordinary bridge's pinned artifact identities |
| Older `prepared` reference package | about 15.5 s | Full exact prepared execution; useful for reference, unsuitable for live updates |
| Accurate primary candidate before later scheduling | 42.014 ms | Starting point of the later accuracy-preserving optimization work |
| Exact-plane/replay candidate | 14.598 ms | Later native arithmetic/staging improvements; superseded as the fastest measured recipe |
| Final recorded `l8` primary recipe | 9.681 ms | Active-block scheduling; 54/54 runs over 27 synthetic cases, plus jitter/adversarial-order evidence |

The `l8` paired smoke measurements were roughly 9.65 ms. The full-case median is the more complete headline. These are depth-engine measurements, not camera-to-display latency or proof of real-scene metric accuracy. The alternate model was not automatically covered by the optimized primary schedule.

## What produced the durable gains

Exact u8 conversion planes, compact staging, reduced pool/cache traffic, and per-command plan replay improved native execution substantially. Splitting worker work into dependency-aware lanes and tightening active memory footprints improved it further. Waiting-thread behavior and unnecessary barriers mattered alongside kernel arithmetic. Read [scheduling](scheduling.md) for mechanisms and rejected changes.

Older peak-GEMM measurements establish HMX capability. Early AXI measurements establish a DDR bottleneck in specific firmware/session states. Neither justifies assuming that the current model is limited by a low clock. The later live probes found both AXI and HMX configured correctly.

## Oracle and accuracy

“Exact” must identify its reference. The tree has used a simulator oracle, native-hardware rounding oracles, and a mixed-rounding accurate candidate. Matching an older hardware oracle did not necessarily meet the later f32 error goal.

The final primary work retained byte identity with its selected accurate oracle over the synthetic case set. Those cases include shifts, flat patterns, ramps, and noise. They do not establish accuracy on real stereo pairs, calibrated uncertainty, room-scale stability, or generalization of a future distilled model.

Standalone legacy-device Conv/Add/Convert and tested MatMul graphs matched their compatible simulator on exercised inputs. Full-model textured-input differences remained unexplained. Do not use the superseded blanket midpoint-rounding explanation as an established result.

## Measuring changes

Use paired alternating control/candidate runs, distinguish first-call setup from warm inference, and retain input/output and binary identities. Profiling can add material overhead. Caller wall time, summed worker cycles, lane busy time, and per-op overlap totals are different quantities and cannot be added indiscriminately.

A schedule model is useful for prioritizing work, not a timing oracle. Active-block scheduling predicted the direction of a measured gain; the later split-merge design predicted a large gain but actually regressed. New synchronization and workspace structures need device timing even after numerical checks pass.

Raw reports, engine binaries, captures, and source snapshots remain local under the ignored build/report locations. The former dated performance diary is recoverable in prior Git history and the external documentation recovery copy. The current recipe and analysis code are [here](../../samples/depth-model/deployment/lanes/).
