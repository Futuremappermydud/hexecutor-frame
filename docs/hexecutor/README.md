# Hexecutor architecture

Hexecutor executes native Hexagon work through the Frame's existing CDSP callback infrastructure. Its main depth path validates a prepared program, binds privately owned memory, runs native handlers, and returns outputs through a host-controlled session. It is not a general-purpose QNN replacement or a loader for arbitrary model containers.

## Host and DSP responsibilities

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| Application IPC | `runtime/scripts/model_depth_bridge.py` | Persistent socket, input/output framing, backend/package identity |
| Session host | `resident_dsp.py`, `frame_transport.py` | Exclusive ownership, image upload, memory checks, commands, cleanup |
| Bootstrap/entry | `qd_graph_boot.c`, loader/linker files | Guarded entry, stack/context setup, execution handoff |
| Prepared integration | `qd_graph_prepared.h` | Pool binding, leases, execution hooks, lane/helper dispatch |
| Reader and runner | `native_program/nprg2_runtime.*`, `prepared_executor.*`, `prepared_runner.*` | Format validation, coverage preflight, dispatch and accounting |
| Kernels | Prepared handlers, `qd_hmx_*`, `blocked_glue/` | Convolution, layout, elementwise, and data movement |

Implementation paths in the table, except the application bridge, are relative to `hexecutor/frame/`.

The fixed loader at `0x87606000` invalidates the payload instruction/data-cache range before entry. Basic callback payloads use the linker-defined region starting at `0x87610000`. The full prepared engine has its own private code/heap allocation contract; do not treat the small callback payload limit as the size of every engine.

## Execution modes

The ordinary learned-depth runtime supports `prepared-qnn` and `prepared`. Both retain Python for session control while running graph operations on the DSP. The old hybrid scheduler, its entry points/builders, and its shipped archives were removed. `fast_depth.py` remains a CPU-only validation reference; low-level transport and kernel utilities remain shared infrastructure.

The reference prepared path executes a serial schedule with optional data-parallel handlers. The optimized primary recipe uses generated run/lane tables and cooperative helpers. A schedule valid for the primary model does not establish alternate-model coverage.

Before execution, validate the image and trusted hashes, bind non-overlapping pools, establish ownership and scratch capacity, and preflight every requested command. A matching opcode name is insufficient: shapes, storage class, quantization, aliases, and worker scratch must fit the selected handler's contract. Unsupported commands reject the requested execution range rather than falling back into an old VTCM scratch path.

Firmware state is borrowed. Loader identity, TLB entries, L2 contents, stacks, vector context, callback registration, clocks, and the eye client all belong to the session's restoration contract. See [operations](operations.md) and [memory layout](memory-layout.md).

Peripheral directories retain focused roles: `frame-dsp-followup/` contains early callback probes; `qurt_boot/` explores boot/runtime behavior; `native_adapter/`, `native_stride2/`, and `native_u16/` investigate shape-specific kernels; `llm/` demonstrates another DSP client. These do not independently expand the prepared executor's supported graph contract.
