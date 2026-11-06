# Prepared InputSlicePad CPU-Prefix Hexagon Simulation Report — 2026-10-31

## Context & Variant Analysis

In Phase 2 of the native runtime plan, the Alternate model prefix reaches command 68 (`seq 118`) and command 69 (`seq 120`).

### Definition and Execution Across Codebase
1. **`frame/resident_program.py` & `frame/resident_schema.py`**:
   - Opcode `k_input_slice_pad` is opcode 2 (`#define QDR_INPUT_SLICE_PAD 2u`).
   - The canonical schema variant string is `*InputSlicePad@FB.s4*3.t.s4*3.`, indexed as variant 2 in `resident_schema.py` and `qd_resident_schema.h`.
   - Repeated listings of identical sequences are transformed by `resident_program.py` into `QDR_NOP` (opcode 1, variant 2) with `duplicate_input_pad = True`.
   - Variant 8 in the ABI schema is `@Fill` (opcode 26). Opcode 8 in the ABI is `k_requant_u8` (`linearclip_simple_rq@CB*2.fi`).
   - Sequence 114 is command 65 (`@SyncOp`, opcode 1, variant 10).
   - In Alternate, command 67 is opcode 8 (seq 113), command 68 is seq 118 (opcode 2, variant 2), and command 69 is seq 120 (opcode 2, variant 2).
2. **`frame/native_program/prepared_program.py` & `compact_v2.py`**:
   - Both commands serialize as `seq=118/120`, `opcode=2`, `variant=2`, `io_index=0`, 7 input refs, 1 output ref.
3. **`frame/qd_graph.c`**:
   - Line 500 implements `k_input_slice_pad`: validates 7 inputs, 1 output, shape-value geometry, and dispatches direct contiguous u8 outputs to `qd_layout_input_slice_pad_direct_u8`.
4. **`tools/htp_graph.py`**:
   - Extracts camera input index `graph_input_index = by_id[inputs[4]]['shape_values'][3]`.

### Differences Between Early (cmds 30..46) and Mid-Prefix (cmds 68..69) InputSlicePad
- **Geometry & Shape**: Identical output shape `[1, 10, 258, 1]` u8 FB and camera input shape `[1, 320, 256, 1]`.
- **Start coordinates**: Cmds 30..46 slice lines 311, 231, 239, 247...; cmd 68 slices line 151; cmd 69 slices line 159.
- **Pool 1 destination**: Cmd 68 writes to pool1 offset 1,740,800 (reusing buffer from cmd 33); cmd 69 writes to offset 2,822,144 (reusing buffer from cmd 34).
- **Execution stage**: Cmds 68 and 69 appear in the interleaved feature-extraction stage following 1x1 convolutions and LinearClip requantization.

## Implementation

Added self-contained module `prepared_slice_pad.h` and `prepared_slice_pad.c`:
- Registers `QDR_INPUT_SLICE_PAD` with Variant 2 and Variant 8 via `nprg2_executor_register`.
- Enforces strict read-only validation:
  - Fail-closed arity: 7 inputs, 1 output, 0 vectors, 0 calls, io_index < 2.
  - Geometry validation: inputs 0, 1, 2, 4, 5, 6 verified as runtime `s4` shape-values; input 3 verified as scalar fill (`ni` or zero `s4`).
  - Output tensor verified as contiguous direct `FB` u8 in pool 1 within `pool1_writable_bytes`.
  - Boundary guards, overflow checks, and strict source/destination non-overlap alias proofs.
- Re-validates before execution in `run_slice_pad` and dispatches to `qd_layout_input_slice_pad_direct_u8`.

## Verification & Hexagon Simulator Proof

1. **CPU Prefix Capture**:
   - Authored `frame/native_program/capture_slice_pad_cpu_prefix.py`.
   - Captured exact state immediately before and after cmd 68 (seq 118) and cmd 69 (seq 120) using `NativeGraph` under CPU emulation.
2. **Simulation Proof**:
   - Authored `frame/native_program/test_prepared_slice_pad.c` and `run_prepared_slice_pad_sim.sh`.
   - Built with Hexagon Tools 19.0.07 `-mv75 -mhvx -mhvx-length=128b -O2 -Wall -Wextra -Werror`.
   - Executed under Qualcomm Hexagon SDK 6.6.0.0 simulator (`hexagon-sim -mv75 --quiet`).
   - Verified:
     - Read-only preflight leaves all 7 pools 100% untouched.
     - Unsupported variant mutation rejects with `NPRG2_EXEC_UNSUPPORTED` before writes.
     - Destination offset out-of-bounds mutation rejects with `NPRG2_EXEC_REJECTED` before writes.
     - Wrong input capacity rejects with `NPRG2_EXEC_REJECTED` before writes.
     - Input/output memory aliasing rejects with `NPRG2_EXEC_ARGUMENT` before writes.
     - Execution produces 100% bit-for-bit exact match against CPU oracle.
     - All non-output pool memory (pools 0, 2..6, and pool 1 outside destination patch) is 100% untouched.

## Provenance & SHA-256 Digests

| File | SHA-256 |
|---|---|
| `frame/native_program/prepared_slice_pad.h` | `3c826d6aed691cc085d4f5a6165477d156acdb4ca84457864e06c8b0bb31b488` |
| `frame/native_program/prepared_slice_pad.c` | `deef1a390288d31aa3a76756e9da58d14b21625527b8c37e5b9c14fba274bc7a` |
| `frame/native_program/capture_slice_pad_cpu_prefix.py` | `f5816812cb12452868e994bc39b5d71795de0b5644476f12950e3a07c5912540` |
| `frame/native_program/test_prepared_slice_pad.c` | `278d56a826d825a9e557aa3b03a2a016eaf5bc87ba02d9baf5548cfacf2068b1` |
| `frame/native_program/run_prepared_slice_pad_sim.sh` | `e772100593ca393fb38091a18a0200aa3fb4ff6efecd773067aba900960657c4` |
| `frame/native_program/fixtures/alternate-smoke-slice-pad-cmd68.ppcpu` | `e41b1414168e74ac8402322157c3d86f5939d094094517719c57a2a615773636` |
| `frame/native_program/fixtures/alternate-smoke-slice-pad-cmd69.ppcpu` | `acd540337fdad4198aaa26af9c1913f58c4ec4d06c4d02b8e412ee23cd32e28e` |
