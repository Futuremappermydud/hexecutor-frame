# Prepared activation DMA opcode 17 — 2026-10-30

Implemented only the actual early `ConvLayer.opt.activations_to_vtcm_dma@CB.cB.` profile. Primary is command 5, sequence 67; alternate is command 7, sequence 58. The handler registers `QDR_COPY_ACTIVATION` opcode 17 / variant 17 from `nprg2_executor_register_preprocess()`.

The fail-closed profile requires one input and one output, no vectors/calls/duplicate padding, the two observed sequences, U8 cB→CB geometry with dimensions `[1,1,1,32]`, max dimensions `[1,8,8,32]`, zero padding, one source block from pool 5, and the exact model-specific output slice/count (primary 52/72, alternate 0/20). It verifies encoded block pools, source 128-byte alignment, destination 2 KiB alignment, full-page bounds, pool1 writable ceiling, pointer overflow, and no source/destination overlap. On this proven tuple, the operation copies exactly one 2,048-byte activation page.

## Validation

The fixture capture ran `NativeGraph` in CPU mode immediately before and after each real operation. Both actual fixtures passed in strict host C11 and Qualcomm Hexagon SDK 6.6.0.0 / Tools 19.0.07 `hexagon-sim -mv75 --quiet` runs:

- Primary command 5 / sequence 67: exact 2,048-byte destination page at pool1 offset 3,756,032; all other bytes in every pool unchanged.
- Alternate command 7 / sequence 58: exact 2,048-byte destination page at pool1 offset 4,018,176; all other bytes in every pool unchanged.
- Read-only preflight passed without changing pool bytes.
- Mutating the command to unsupported variant 18 returned `NPRG2_EXEC_UNSUPPORTED` before writes.
- Redirecting the output block to pool1 offset `0x780000` returned `NPRG2_EXEC_REJECTED` before writes.

An initial test rejected the actual source offset because it required 2 KiB alignment. The source block is 128-byte aligned at pool5 offset 17,152 (primary) / 17,408 (alternate); the operation reads a 2 KiB cB tile there. The final guard requires 128-byte source alignment and 2 KiB destination alignment, matching the serialized layout.

Reproduce both fixture capture and SDK simulation with:

```sh
sh frame/native_program/run_prepared_activation_dma_sim.sh
```

## SHA-256 provenance

- `prepared_preprocess.c`: `ae788df3c8f06ffb7234b3faa732550bdfb4ce136966a4a28f58db4ce04ef3fe`
- `prepared_preprocess.h`: `3b808d85c6d61d7d1546cbecb859c6240130da7c6b9c43752260376e86f69a13`
- `capture_activation_dma_cpu_prefix.py`: `d4caf9b957ad5df9f2c5671079ff822e697ce9f4c84cf9e27a215635076f7cbc`
- `test_prepared_activation_dma.c`: `0308e47650f7831a88f6629309606746d1f11481bb49375e2f2283af91fd3c32`
- `run_prepared_activation_dma_sim.sh`: `e7b0830b11f3ab7ad9d314af4c05e58046515849e25cfacee976c71bbea2f10e`
- primary fixture: `78e881d8b4a410acbb2f9e09d46bd219f4979955fe535f74c76966d9d49f39d5`
- alternate fixture: `31f2fb0b6e049c81fa75797c90fbdad2268ea2984f1f3b49ec2b53ff486820d9`

No root-owned files or device were changed. This bounded opcode proof does not by itself claim full-model execution support.
