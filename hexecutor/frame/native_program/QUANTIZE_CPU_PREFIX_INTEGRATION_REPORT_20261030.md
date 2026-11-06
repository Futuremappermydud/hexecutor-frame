# Prepared Quantize integration — 2026-10-30

Integrated the independently proved `Quantize@FH.Ff` profile as opcode 28 / variant ID 48. Registration is opt-in: `nprg2_executor_register_preprocess()` registers it only when `QDS_PREPARED_QUANTIZE=1`; default-off builds retain existing coverage. Root build integration needs the two new C files `prepared_quantize.c` and `prepared_quantize_core.c` when that flag is enabled. Headers are included from native_program. `prepared_preprocess.c` now also registers the already-proved activation DMA handler.

## Exact accepted profile

| Model | command / sequence | direct pool1 input → output | zero point / output scale |
|---|---:|---:|---|
| primary | 60 / 107 | 1,089,536 → 7,589,888 | 32768 / `0.0003051804378628731` |
| alternate | 54 / 100 | 3,997,696 → 2,775,040 | 32768 / `0.0003051804378628731` |

Both use Ff float32 → FH uint16, dimensions/max `[1,2,4,4]`, zero padding, 128-byte input and 64-byte output, disjoint direct storage. The accepted output scale rounds to float32 bits `0x39a000a0`. The handler checks the model-bound command index and sequence, all tensor metadata/offsets, the scale and zero point, writable bounds, overflow and aliasing.

The command call record must be exactly call ID 12, pointer mask `0x85`, and ten argument words:

`[0, 32768, 0, 32, 0, 0, 1, 0, 0x454ccc00ff577600, model_arg9]`

Argument slots 0, 2 and 7 are masked pointers. `model_arg9` is `0x5555560d0500` for primary and `0x5555560d0700` for alternate. Every word is validated before execution.

## Numerical and pool results

Strict host C11 and Hexagon SDK 6.6.0.0 / Tools 19.0.07 simulator runs passed both actual NativeGraph CPU-prefix fixtures. Both produced the exact 32-value, 64-byte output. Pool1 matched its full 8 MiB CPU preimage outside the output interval, and all other pools remained unchanged. Read-only preflight left pools untouched. Wrong variant, altered call arguments and aliased output metadata each failed before graph-pool writes.

A NaN inserted into the source made the bounded one-command run return handler error before writing the destination. Source tensor 66 / 54 is produced by command 0 (`InputSlice@Ff`) in the respective graphs. Therefore the Quantize validator does not claim that inspecting current pool1 values during preflight proves what a preceding command in a multi-command range will later produce. This proof covers the bounded Quantize command and its no-output-write response to nonfinite input; it does not claim whole-prefix atomicity after earlier commands have run.

The numerical core preserves the CPU calculation: float32-rounded reciprocal, double multiply and `+0.5`, then clamp. It implements positive `floor` by clamp-before-integer-conversion and checks float32 finiteness by exponent bits, without `floor`/`isfinite` library calls. An SDK `-nostdlib -ffreestanding` object build succeeded and `hexagon-nm -u` reported no undefined symbols.

## Reproduction

- Host strict C11 fixture test: compile `test_prepared_quantize.c`, `prepared_quantize.c`, `prepared_quantize_core.c`, `prepared_preprocess.c`, `prepared_executor.c`, and `nprg2_runtime.c` with `-DQDS_PREPARED_QUANTIZE=1 -Wall -Wextra -Werror -pedantic`; link with the host C runtime and run the binary.
- SDK fixture proof: `sh frame/native_program/run_prepared_quantize_sim.sh`. This uses the existing manifest-bound fixtures under `frame/native_quantize/fixtures/` without rewriting them.

## SHA-256 provenance

Changed/native_program implementation and proof files:

- `prepared_quantize.c`: `8c6e48769251a6b667ff06b0b9e85938b2d44ae84582593d1517e3c6fdc8997f`
- `prepared_quantize.h`: `6ba044060f7987cc95af75c1e03a1e652b8a4bd5fbedc0604a93718918e60271`
- `prepared_quantize_core.c`: `d5283a6bdc6cc1e52ca428fb81bea8959e0fc2295ec6dd6cdf08916c863b3e78`
- `prepared_quantize_core.h`: `53f54a2a60f04722e791c43315185967f73b94468d98f530d9774e6fb5940b5c`
- `prepared_preprocess.c`: `3f5a066acd789728a7e17aefbb8087da98d3b721c36eccd09a8646cb896957b1`
- `test_prepared_quantize.c`: `224166f58d03d47e48fc6d45888a7092f987af903a67db4a6d164af09ff4e950`
- `run_prepared_quantize_sim.sh`: `52da5659767ba500d7c7e6ca6e63d260ab64f302a00c1dc4c84f3b0485da25ee`

Fixture manifests and payload hashes are the ones recorded in `frame/native_quantize/fixtures/primary-smoke.json` and `alternate-smoke.json`; source and expected output hashes match across models: source `59160118b6ec92c768dbf108f969dc12d786fa19935201e3a3755ee1a000ff74`, output `95ada9194ff9e8a8f7d00604f6b2b7a20f2c0a26703bae5f120f7920bdd40e86`.
