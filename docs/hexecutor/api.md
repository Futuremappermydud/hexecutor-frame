# Native host API

The live inference path uses the C++17 API in `hexecutor/include/hexecutor/executor.hpp`. Python modules are retained for offline extraction, lowering, reference comparisons, and diagnostic tools; the headset depth application does not start a Python interpreter.

`TensorSpec` declares a name, scalar type, and dimensions with overflow-checked byte sizing. `Program` declares an image, constant pool, and ordered input/output contracts. `Artifacts` supplies explicit loader, engine, bootstrap, memory-map probe, prepared-program, and firmware identities. `Session` owns the verified device session and accepts buffers with exact byte lengths.

The existing QDRX v1 wire ABI supports four inputs and two outputs. The separate [tensor backend](tensor-backend.md) supports variable float32 I/O and multiple graphs under one device lease via `Session(std::vector<Program>, Artifacts)` and `run(buffers, program_index)`. That is an explicit backend constraint, not a claim that arbitrary neural graphs are supported. Tensor semantics and shapes belong to the model package, while native handlers and program preflight still decide which operations are supported.

The depth adapter is separate:

```cpp
#include "depth_model/deployment.hpp"

auto deployment = depth_model::load_deployment(package_path, "prepared-qnn", "primary");
hexecutor::Session session(deployment.program, deployment.artifacts);
// Each Buffer supplies a pointer and exact byte length in declared input order.
auto outputs = session.run(input_buffers);
session.close();
```

`load_deployment` owns the depth shapes, artifact identities, and model selection. It rejects unsupported model/backend combinations and verifies pinned files before device access. `FRAME_DEPTH_PACKAGE` is the runtime's package selection; `FRAME_DEPTH_ARTIFACTS`, `FRAME_DEPTH_PROGRAMS`, and `FRAME_DEPTH_CHUNK_PROBE` select separately provisioned locations. A legacy installed package may have its own physical `frame/` directory; the source tree no longer uses symlinks to emulate that layout.

The host ports the firmware hash/ELF checks, shared-descriptor handshake, exclusive lock, eye-client pause/restart, chunk-map verification, private allocation guards, prepared dispatch, and VTCM/clock restoration checks. Cleanup retains callback data and leaves eye tracking paused if restoration cannot be proved. `Artifacts::keep_eye_running` is reserved for `hexecutord`'s Gazenet adapter, which services or explicitly yields the stock eye requests; ordinary sessions must retain the default pause behavior. `Artifacts::max_wake_seconds` (default 1 s) bounds how long one firmware wake is held: the transport lets the workers exit and wakes them again before a command that would overrun it, and a watchdog thread does the same during idle gaps. Every such exit gets the same worker, JTLB and VTCM checks as close. The first run after a re-wake re-stages resident records, so it is slower; completion prediction ignores it, and also ignores the first run after a graph switch. Destruction attempts cleanup, but callers should invoke `close()` explicitly to receive a cleanup error.

`hexecutor/broker.hpp` adds a [CDSP broker](broker.md) client and server: `hexecutord` owns one tensor session and clients submit jobs with `broker::Client::run`, which has the same shape as `Session::run`.

The native worker is `build/bin/frame_depth_worker` (copied to `runtime/bin/` for the app). `FRAME_DEPTH_WORKER` overrides its executable. The existing protocol v3/v4 is retained; `--check-package` verifies provisioned package files without opening the DSP. The worker supports idle close/reopen and close-after-reply. The portable protocol tests inject an inference implementation rather than adding a fake backend to the production worker.

Build with `make build-host`; run the portable native checks with `make check`. The Linux transport is compile-checked separately from macOS's hardware-unavailable branch. These checks do not substitute for a real headset inference and restoration run.
