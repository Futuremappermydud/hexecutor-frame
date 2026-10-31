# Prepared activation DMA and Quantize bounded Frame validation

The opt-in engine `68b81b3c3ca5d0fadd19483986b1566ae8dd7f48de045c34efe450ab0449a22d` was built from the 92-file frozen source tree `.work/frame-direct-kernels-20261030/native-prepared-quantize-first-source/`. Source hashes and `BUILD_FLAGS.json` were captured before Docker compilation. The preserving bootstrap hash was `628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147`.

The new `QDS_PREPARED_QUANTIZE=1` flag enables Quantize registration and gives the ELF a distinct `-quantize` name. Activation DMA is in the prepared preprocess handler. Actual CPU-prefix host and SDK fixtures passed for both operations before Frame execution; see `native_program/ACTIVATION_DMA_CPU_PREFIX_REPORT_20261030.md` and `native_program/QUANTIZE_CPU_PREFIX_INTEGRATION_REPORT_20261030.md`.

| Model | Bounded command | Cases | Result |
| --- | --- | --- | --- |
| Primary | activation DMA `[5,6)` | smoke, random | exact in pools 1, 3, 4, 5, 6 |
| Primary | Quantize `[60,61)` | smoke, random | exact in pools 1, 3, 4, 5, 6 |
| Alternate | activation DMA `[7,8)` | smoke, random | exact in pools 1, 3, 4, 5, 6 |
| Alternate | Quantize `[54,55)` | smoke, random | exact in pools 1, 3, 4, 5, 6 |

The same ELF passed sampled smoke regressions for primary/alternate padded 1x1 sequences 115/108 and staged 3x3 sequences 247/180. Those convolution outputs stayed within the previously accepted 0/+1 CPU bound, and all outside-target pools matched. Each of the eight Frame sessions exited with `[0,0,0]` errors, lease phase 2, no lease error, and equal independent host VTCM fingerprints. All runs were bounded ranges with boundary pool transfers; none establishes full-model latency.

Quantize's input is produced by command 0 in both models. Its finite-value check occurs immediately before the target command's destination write. This test does not establish that a longer range containing earlier writes rejects a future nonfinite Quantize input before those writes. Full-prefix atomicity and whole-model prepared execution remain unsupported.

The eight JSON reports are `.work/frame-direct-kernels-20261030/native-prepared-quantize-{primary,alternate}-{activation,quantize,padded-regression,3x3-regression}.json`; their hashes are recorded in the frozen source tree's `PROVENANCE.json`. After testing, `cdsp_mem` was unloaded. Final read-only state: CTRL `[1,0,1]`, callback zero, loader/payload/arena zero, eye client parked in `S` state.
