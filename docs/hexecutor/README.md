# Hexecutor architecture

Hexecutor executes native Hexagon work through the Frame's existing CDSP callback infrastructure. Its main depth path validates a prepared program, binds privately owned memory, runs native handlers, and returns outputs through a host-controlled session. It is not a general-purpose QNN replacement or a loader for arbitrary model containers. The separate [generic tensor backend](tensor-backend.md) supports explicitly lowered static graphs and shared multi-model sessions, including [MediaPipe hands](../model-development/hand-tracking.md).

## Host and DSP responsibilities

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| Application IPC | `depth-runtime/tools/frame_depth_worker.cpp` | Persistent native socket protocol and lifecycle |
| Model adapter | `samples/depth-model/deployment/cpp/package.cpp` | Depth contract, package identities, and model selection |
| Executor host | `hexecutor/host/cpp/` | Generic tensors/programs, transport, private memory, dispatch and cleanup |
| Bootstrap/entry | `hexecutor/native/qd_graph_boot.c` and loader/linker files | Guarded entry and stack/context setup |
| Prepared integration | `hexecutor/native/qd_graph_prepared.h` | Pool binding, leases, execution and helpers |
| Reader and runner | `hexecutor/native/program/` | Format validation, preflight, dispatch and accounting |
| Kernels | `hexecutor/native/kernels/` and native HMX headers | Convolution, layout, elementwise and data movement |


Paths in the table are relative to the repository root. Read [the native API guide](api.md) for embedding the host. The [CDSP broker](broker.md) schedules supported tensor clients in one native process. Prepared-depth and QNN16 workloads retain their own engine and session contracts.

The fixed loader at `0x87606000` invalidates the payload instruction/data-cache range before entry. Basic callback payloads use the linker-defined region starting at `0x87610000`. The full prepared engine has its own private code/heap allocation contract; do not treat the small callback payload limit as the size of every engine.

## Execution modes

The ordinary learned-depth runtime supports `prepared-qnn` and `prepared`. Both use the native C++ host for session control and run graph operations on the DSP. The old hybrid scheduler, its entry points/builders, and its shipped archives were removed. `depth_model.reference.native_reference` remains a CPU-only validation reference; low-level transport and kernel utilities remain shared infrastructure.

The reference prepared path executes a serial schedule with optional data-parallel handlers. The optimized primary recipe uses generated run/lane tables and cooperative helpers. A schedule valid for the primary model does not establish alternate-model coverage.

Before execution, validate the image and trusted hashes, bind non-overlapping pools, establish ownership and scratch capacity, and preflight every requested command. A matching opcode name is insufficient: shapes, storage class, quantization, aliases, and worker scratch must fit the selected handler's contract. Unsupported commands reject the requested execution range rather than falling back into an old VTCM scratch path.

Firmware state is borrowed. Loader identity, TLB entries, L2 contents, stacks, vector context, callback registration, clocks, and the eye client all belong to the session's restoration contract. See [operations](operations.md), [memory layout](memory-layout.md), and [dynamic memory](dynamic-memory.md).

Peripheral directories retain focused roles: `research/bringup/` contains early callback probes; `research/qurt-boot/` explores boot/runtime behavior; `native/kernels/adapters/` and `native/kernels/stride2/` investigate shape-specific kernels; `examples/llm/` demonstrates another DSP client. These do not independently expand the prepared executor's supported graph contract.

## License

hexecutor is under the GNU Lesser General Public License, version 3 or later (`LGPL-3.0-or-later`); the kernel
modules in `hexecutor/device/kmod/` stay `GPL-2.0`. Programs under other licenses can use it as a dynamically
linked library (`libhexecutor_et.so`, see [the ExecuTorch delegate](../../hexecutor/executorch/README.md)). Terms,
exceptions and what a distributor must ship: [hexecutor/LICENSE.md](../../hexecutor/LICENSE.md).
