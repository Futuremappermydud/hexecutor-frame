# CDSP broker

`hexecutord` is the native C++ daemon for sharing one HXFG tensor session or one Q16P session. It serves frame hand trackers, dynamically registered graphs such as ExecuTorch delegates, and optional XRService adapters. The application depth worker uses its own prepared QDRX/NPRG engine. A combined prepared-depth and tensor engine would require a shared dispatch and private-memory layout. [Native API](api.md), [XRService bridge](xrservice-broker-bridge.md)

## Client and scheduling contract

- Clients connect on a Unix stream socket and send length-prefixed messages with inline input/output bytes. A client opens a `StreamSpec` against a registered graph and can use `submit`/`wait` or synchronous `run`. The public API is in [broker.hpp](../../hexecutor/include/hexecutor/broker.hpp).
- The daemon preflights startup `--graph NAME=IMAGE,CONSTANTS` programs and supports HXFG runtime registration by image and constant identity. At most eight graphs fit one session. The device session opens on the first job and can be rebuilt when a graph is added. Verify stateful-client recovery across registration. [Daemon](../../hexecutor/host/cpp/hexecutord.cpp)
- One executor runs jobs by earliest deadline, with tracking, interactive and bulk priority used as a tie-breaker. It expires queued jobs before dispatch and can replace an older queued job for a latest-only stream. Graphs run to completion. Admission uses measured in-process cost or the client's cost hint and the configured utilization limit (default 0.7). [Scheduler](../../hexecutor/host/cpp/broker.cpp)
- `--idle-close-ms` defaults to 2000. Idle close can release an ordinary tensor session. Adapter holds and persistent-state graphs keep it open. A disconnect drops that client's queued work and streams. Fatal device errors trigger restoration and leave the daemon refusing further work. [Failure handling](operations.md#failure-handling)

## Q16P sessions

Pass `--qnn16-graph NAME=PROGRAM.q16p,CONSTANTS.q16c` for each startup graph and supply the QNN16 engine with the preserving four-worker tensor bootstrap. A Q16P session accepts only startup Q16P graphs. XRService adapters and dynamic registration require an HXFG session.

When the constants exceed the low DDR region, build [the QNN16 engine](../../hexecutor/build-scripts/build_qnn16.sh), [the tensor bootstrap, and its chunk probe](../../hexecutor/build-scripts/build_tensor.sh) with the same `HEXECUTOR_QDO_SLOTS`. Each slot maps 16 MiB. Set `HEXECUTOR_CHUNK_WINDOW_MB` to the matching size at launch. The Q16P session checks the artifact slot counts, requests and pins a lease covering that window, and then runs the probe. It can place constants at or above 128 MiB. The fixed private QNN16 scratch region is just under 8 MiB, and all graphs share just under 16 MiB for float32 exchange buffers. These limits are independent of the DDR window. A Q16P session skips automatic lease shrink when device cleanup cannot be verified.

## Persistent state

The session stores KV and causal convolution history in a dedicated DDR region. Compatible tensors may share storage by nonzero name hash. Routine wake rotation retains the mapped lease. A forced release records state loss and subsequent runs require a fresh state reset. `Client::reset_state()` is global to the daemon and clears all resident state; a fatal transport error also requires recovery of the failed daemon/session. Server stats include per-graph `state_bytes`. [Tensor state contract](tensor-backend.md#persistent-state-and-decode)

## Recorded measurements and wake policy

### Frame results, 2026-11-09

Hand tracker on a two-hand test image through `hexecutord` versus its own `Session`: outputs identical; 5.21 → 6.10 ms mean per frame. `hexecutor_broker_bench` puts the broker's own IPC overhead at 0.11–0.26 ms per job; the rest of the gap is per-job host work inside `Session::run`.

Shared resident records (`hx_fp16_share` in `native/tensor/fp16.c`, staged bitmask in `engine.c`) against the validated planar7 engine, both through the broker with the validated four-worker bootstrap. DSP time per job:

| Mix | Graph | planar7 | Shared records |
| --- | --- | ---: | ---: |
| landmark ↔ FeatureNN | landmark / FeatureNN | 2.36 / 1.48 ms | 1.20 / 0.59 ms |
| palm ↔ landmark | palm / landmark | 3.19 / 2.45 ms | 2.31 / 1.03 ms |
| landmark only, FeatureNN idle | landmark | 0.96 ms | 1.19 ms |

The alternating workload benefited from shared residency. An idle registered graph reduced the residency budget for the only active graph. The three-graph limit reported in that run applied to its old constant layout. For a new mix, preflight the actual pools with the matched engine and lease. [Tensor memory contract](tensor-backend.md#format-and-execution-contract)

### Wake rotation and the XRService tripwire

The once-a-second ~43 ms stall is the 1 s wake rotation. `HEXECUTOR_WAKE_TRACE` on a 1,000-job landmark run showed one rotation: workers exit 16.1 ms (VTCM restore 11.4 M cycles), host check 4.5 ms, re-wake 3.7 ms arrival plus 18.9 ms setup.

`hexecutord --long-wake SECONDS` raises the wake limit only when XRService keeps controller FeatureNN off the CDSP: the settings file must contain `FeatureNN.enableForLogic: false`, and the running XRService must have started after the file was last written. A watcher then follows `journalctl -t XRService`. Any line containing `cDSP`/`CDSP`, or an XRService restart, trips it. The daemon releases the device at once (`Server::release_device()`) and uses 1 s rotation for the rest of its life. On the Frame (2026-11-09), `--long-wake 30` ran 3,000 landmark jobs with no rotation and a 2.13 ms worst case. An injected `logger -t XRService "[CDSPRunner] …"` line released the device within 35 ms, the next job reopened the session in 410 ms, and rotations resumed. The tripwire reacts after XRService has logged; it limits exposure but cannot stop a dispatch that has already happened. A worn soak with a controller in use has not been done.

## Device validation

Measure camera-to-result time, deadline misses, and cleanup with the exact registered graph set. Run a sustained worn-session test with controller/gaze inputs for any enabled stock adapter. Confirm whether repeated graph switches exhaust the chosen VTCM residency budget before proposing admission rates. A combined prepared-depth engine, graph preemption and memfd client buffers remain design work.
