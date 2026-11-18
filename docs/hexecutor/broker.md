# CDSP broker (proposed)

Status, 2026-11-09: phase 1 and the FeatureNN/Gazenet stock-job adapters are implemented with portable ABI tests; both still require on-device tracking validation. The combined depth engine remains a proposal.

## Phase 1 as built

- `hexecutor/include/hexecutor/broker.hpp`, `hexecutor/host/cpp/broker.cpp`: scheduler, admission, Unix-socket server and `Client`. Frames are length-prefixed over a stream socket; inputs travel inline (memfd zero-copy is not done).
- `hexecutor/host/cpp/hexecutord.cpp` (`build/bin/hexecutord`): wraps one tensor `Session` over up to eight `--graph NAME=IMAGE,CONSTANTS` graphs. It opens the session on the first job and closes it after `--idle-close-ms` (default 2000) with nothing queued. After any device error it refuses further work rather than reopening.
- `runtime/scripts/run_hexecutord.sh` starts it with the hand graphs registered as `palm` and `landmark`; extra `--graph` arguments are passed through.
- Clients: `frame_hand_tracker --broker SOCKET frame.ppm …` (replay, every frame runs) and `frame_hand_viewer --broker SOCKET` (latest-only, deadline two frame periods).
- `hexecutor/tests/test_broker.cpp` covers ordering, replacement, expiry, admission, contract checks, idle release and stats with a fake backend over a real socket.

### Frame results, 2026-11-09

Hand tracker on a two-hand test image through `hexecutord` versus its own `Session`: outputs identical; 5.21 → 6.10 ms mean per frame. `hexecutor_broker_bench` puts the broker's own IPC overhead at 0.11–0.26 ms per job; the rest of the gap is per-job host work inside `Session::run`.

Shared resident records (`hx_fp16_share` in `native/tensor/fp16.c`, staged bitmask in `engine.c`) against the validated planar7 engine, both through the broker with the validated four-worker bootstrap. DSP time per job:

| Mix | Graph | planar7 | Shared records |
| --- | --- | ---: | ---: |
| landmark ↔ FeatureNN | landmark / FeatureNN | 2.36 / 1.48 ms | 1.20 / 0.59 ms |
| palm ↔ landmark | palm / landmark | 3.19 / 2.45 ms | 2.31 / 1.03 ms |
| landmark only, FeatureNN idle | landmark | 0.96 ms | 1.19 ms |

Alternating graphs no longer re-stages; a session where one registered graph never runs pays for the split. The engine should fall back to one graph's standalone plan when only that graph is running. Palm, landmark and FeatureNN together (37 MB of constants) exceed the tensor constant region (32–64 MiB of the lease), so the host refuses that session before touching the device. Pairs fit.

### Wake rotation and the XRService tripwire

The once-a-second ~43 ms stall is the 1 s wake rotation. `HEXECUTOR_WAKE_TRACE` on a 1,000-job landmark run showed one rotation: workers exit 16.1 ms (VTCM restore 11.4 M cycles), host check 4.5 ms, re-wake 3.7 ms arrival plus 18.9 ms setup.

`hexecutord --long-wake SECONDS` raises the wake limit only when XRService keeps controller FeatureNN off the CDSP: the settings file must contain `FeatureNN.enableForLogic: false`, and the running XRService must have started after the file was last written. A watcher then follows `journalctl -t XRService`. Any line containing `cDSP`/`CDSP`, or an XRService restart, trips it. The daemon releases the device at once (`Server::release_device()`) and uses 1 s rotation for the rest of its life. On the Frame (2026-11-09), `--long-wake 30` ran 3,000 landmark jobs with no rotation and a 2.13 ms worst case. An injected `logger -t XRService "[CDSPRunner] …"` line released the device within 35 ms, the next job reopened the session in 410 ms, and rotations resumed. The tripwire reacts after XRService has logged; it limits exposure but cannot stop a dispatch that has already happened. A worn soak with a controller in use has not been done.

Not yet built: rate-aware VTCM budgets (records are shared, not yet weighted by use), the static cycle, measured-cost admission across restarts, and memfd inputs.

Several DSP workloads now compete for one CDSP: learned depth, MediaPipe hands, XRService's controller FeatureNN and eye tracking, and later our own eye model. Today each of our clients opens its own exclusive `hexecutor::Session`, and XRService dispatches its own jobs through `dsp_service` whenever it likes. That arrangement is what left the firmware at `(2,0,8)` (see [operations](operations.md#xrservice-controller-featurenn-and-continuous-wakes)). It also pays a full wake, VTCM backup and weight staging every time ownership changes hands.

The broker makes one host process the only owner of the CDSP. Every DSP job, ours and eventually XRService's, goes through its queue.

## Constraints from current measurements

| Fact | Source | Consequence |
| --- | --- | --- |
| One exit plus wake costs about 43 ms; the first result after a wake arrives about 22 ms in. | Wake trace, 2026-11-08 | Ownership cannot change per frame. Hold one wake across many jobs. |
| A preserving wake copies 8 MiB of VTCM out and reads about 32 MiB of DDR around it. | [Re-wake traffic](operations.md#re-wake-traffic) | Same as above. |
| XRService dispatching FeatureNN during our wake leaves `(2,0,8)` until reboot. | [Controlled test](operations.md#controlled-cpu-versus-cdsp-test-2026-11-09) | Until the broker serves XRService's jobs, XRService must be kept off the CDSP with the `controllerTrackingOptions` setting. |
| Depth (prepared QNN lanes): 9.68 ms exact. Hand landmark 0.95 ms, palm 2.2 ms. | [Scheduling](scheduling.md), hand checkpoint | Depth is the only long job; tracking jobs are about 1–3 ms. |
| FeatureNN lowers to a 68-op HXFG graph: 3.9 MB arena, 5.8 MB resident HMX weights in the FP16 plan, 2.79 M serial simulator cycles (74 % depthwise). | `hexecutor/offline/lower_onnx.py`, 2026-11-09 | It fits the tensor backend. Device time is unmeasured. |
| `Session` already runs up to eight tensor graphs under one lease (`run(buffers, program_index)`). | [Tensor backend](tensor-backend.md) | Hands, FeatureNN and an eye model can share one session today. |
| Depth uses the QDRX engine; tensor graphs use the tensor engine. Their private-DDR maps overlap (both start at relative 0). | [Memory layout](memory-layout.md), [tensor backend](tensor-backend.md#device-memory-and-execution) | Depth and tensor graphs cannot share a wake without a combined engine and a unified memory map. |
| Depth's VTCM pool 1 is the full 8 MiB, but it holds no live state between inferences. | [Memory layout](memory-layout.md#tensor-pools-in-the-primary-workload) | Tracking jobs can use VTCM between depth inferences. Interrupting depth mid-inference needs a proved low-liveness split point. |
| A tensor graph re-stages its resident weights after any graph switch. | [Tensor backend](tensor-backend.md#device-memory-and-execution) | Alternating hands and FeatureNN every frame would re-stage about 10 MB per cycle unless VTCM is partitioned between them. |

## Architecture

```
 depth worker ─┐                      ┌── hexecutord (one process, C++17) ────────────────┐
 hand tracker ─┼─ unix socket + memfd ─┤  admission → scheduler → one Session/lease → CDSP │
 controller    │   (BrokerClient)     │                      ▲                            │
 eye model    ─┘                      │  stock FeatureNN/Gazenet job adapter               │
                                      └────────────────────────────────────────────────────┘
```

- **`hexecutord`** is a native C++17 daemon. It owns the only `Session`, performs all restoration, and is the single place that decides when to wake, run, rotate or close. No Python runs in it.
- **Clients** link a small `BrokerClient` library. Each client gets a shared-memory region (memfd) with a few input/output slots, so camera-sized inputs are not copied through the socket. The broker still copies inputs into the private exchange arena, as `Session::run` already does.
- **Models** are registered once per broker lifetime. Identical images (same HXFG/constant SHA-256) are shared between clients. The broker preflights every program before it opens the device, exactly as `Session` does now.

### Client API sketch

```cpp
namespace hexecutor::broker {
enum class Class { tracking, interactive, bulk };       // FeatureNN/eye, hands, depth
struct Stream {                                          // one periodic input source
    std::string name;
    Class cls;
    double period_ms;          // expected arrival period, used for admission
    double deadline_ms;        // relative to submit; a stale result is dropped, not returned late
    bool latest_only = true;   // a newer submit replaces a queued older one
};
class Client {
  public:
    static Client connect(const std::filesystem::path &socket, const std::string &name);
    Model load(const std::vector<Program> &programs);     // validated by the broker
    StreamHandle open(Model, size_t program_index, const Stream &);
    Ticket submit(StreamHandle, const std::vector<Buffer> &inputs, uint64_t capture_ns);
    std::optional<Result> poll(Ticket);                   // outputs, DSP timing, or dropped/expired
    Result wait(Ticket);
    // Drop-in replacement for Session::run for existing callers.
    std::vector<Bytes> run(StreamHandle, const std::vector<Buffer> &inputs);
};
}
```

`run()` lets the hand tracker and depth worker switch from an owned `Session` to the broker with a one-line change. `latest_only` matches what camera pipelines actually want: when a new frame arrives before the old one ran, the old one is worthless.

## Scheduling

The workloads are periodic with known costs, so a static cycle plus earliest-deadline-first for sporadic work fits better than a general priority queue.

1. **Cycle.** Per 33.3 ms (one depth frame at 30 Hz): one depth inference (about 10 ms) at a fixed offset, and the tracking and interactive slots placed around it so that 60 Hz hands and per-frame FeatureNN never wait behind depth for more than one slot.
2. **Within free time**, run ready jobs by earliest deadline, with `tracking` ahead of `interactive` ahead of `bulk` on ties. Palm detection (2.2 ms, only on track loss or every tenth frame) is the main sporadic job.
3. **No preemption inside a graph.** Tensor graphs join every operation and keep their state in VTCM, so stopping one midway would mean saving that state. Tracking graphs are short enough not to need it. If depth's 10 ms blocking proves too long, split its run list at a point where little VTCM is live, using the existing liveness analysis (`depth-model/deployment/lanes/mem_layout.py`).
4. **Graph switches cost re-staging.** Group consecutive jobs of the same graph. Give each co-scheduled tensor graph a fixed VTCM budget for resident weights instead of letting each one fill VTCM greedily. Today the FP16 planner fills whatever is free per graph; it needs a per-graph cap.
5. **Admission.** A stream is admitted only if the sum of `cost / period` over admitted streams stays under a utilization limit, about 70 % to start. Costs come from measured device timings that the broker records per graph, not from simulator cycles.

A rough budget at 30 Hz depth, 60 Hz hands and two 60 Hz controller streams: 290 ms depth + about 60–180 ms hands + FeatureNN at an unmeasured 1–2 ms × 120 per second. That is roughly half the DSP. The measured device costs must confirm it before any stream is admitted at those rates.

## Wake policy

The broker holds one wake while it has admitted streams and closes after an idle timeout. With XRService kept off the CDSP, the controlled test held single wakes for 20–60 s without trouble. A longer hold is still unproven. The broker therefore keeps the existing rotation as a configurable safety valve, logs every exit/wake with its duration, and can raise the interval once long holds are demonstrated. A rotation is scheduled between jobs, never inside one, and it counts against the cycle like any other job.

## Engines and memory

- **Phase 1 (tensor-only).** Hands, FeatureNN and a future eye model all use the tensor engine. One `Session` with several programs already supports this. Only the broker, client library and per-graph VTCM budgets are new.
- **Phase 3 (depth).** Depth and tensor graphs need one engine image whose dispatcher handles both the QDRX prepared executor and the HXFG backend. The private-DDR map must be unified so the two layouts no longer overlap: for example depth in relative 0–48 MiB and tensor regions above it. That may need a larger `cdsp_mem` lease or more 16 MiB TLB mappings. Swapping engines inside a wake is the fallback. It needs the loader's instruction-cache maintenance and a re-staging plan, and it costs more per switch.

## XRService jobs

`xrservice::Adapter` connects to `dsp_service`, maps the same stock DMA buffer, and uses ordinary broker `Client` streams; no process hook, binary patch, `LD_PRELOAD`, or second `Session` is involved. It warms and pins the tensor session, submits adapted work with tracking class and a 50 ms deadline, and stages every output before publishing completion. One adapter thread coordinates FeatureNN and Gazenet so they cannot compete for ownership. Weight loading, buffer acquisition, model information, benchmarks and unsupported requests release the session for stock firmware; the adapter reacquires and warms all registered graphs afterwards.

### FeatureNN

`hexecutord --xrservice-featurenn GRAPH` serves job 8 from the unmodified built-in XRService. Its shared block starts at DMA offset `0x1522000`. Its words are state, active, and taken/job id at offsets 0, 4, and 8. The request header is at cached offset `0x80`, packed u8 eye planes at `0x100`, and results at `0x40080`. Each eye has a 0x450-byte `cdsp_featurenn_output_data`: four contiguous 69-float arrays (center, uncertainty, plus-x, plus-z). The adapter converts u8 planes to normalized three-channel float NHWC and writes uncertainty as `2·min(raw,10)`; XRService applies the final exponential.

Enable XRService's normal FeatureNN path (not the GripNet/CPU test setting) and run, using the actual lowered prefix:

```sh
FRAME_FEATURENN=/home/steamos/hexecutor-broker/featurenn \
    runtime/scripts/run_hexecutord.sh --long-wake 30
```

That launcher registers FeatureNN alone; add `--graph landmark=...` explicitly if desired. Palm, landmark and FeatureNN together still exceed the current tensor constant region. The CPU ONNX model that ships with XRService is the numerical reference. Our lowered graph differs from its u8 result by up to about 0.07 on the raw head.

### Gazenet

The stock `CDSPGazenet` library is used by the separate `eyetracking` process. Its control block starts at DMA offset `0x445000`. Job 2 loads weights and the first job 4 acquires the stock I/O buffer; both remain on stock firmware. Once that buffer exists, `--xrservice-gazenet GRAPH` serves job-4 evaluation. The graph contract is one 128×128 three-channel float NHWC input and exactly 64 output floats. The adapter converts the two planar stock float images at cached offset `0x100`, pads channel two with zero, and writes the stock result at `0x20100`.

`--xrservice-gazenet-vector GRAPH` separately adapts the in-place float-vector request. Its input is exactly 4,800 floats at cached offset `0x440`; the graph may return one or several outputs containing 1–4,800 finite floats in total. The adapter writes them back to the same buffer, updates the float count, then clears the ready word at control offset `0x0c`. Other vector sizes are left to stock firmware.

**VTCM hazard (open, 2026-11-17).** Firmware hardware thread 3 runs the stock eye-tracking handler on its own, separately from our wake. It serves jobs 2/4 posted to the DMA block at `0x1522000`, and it uses VTCM `0x31700000–~0x317d4600`. Pool 1 owns that range while a session is open ([memory layout](memory-layout.md#stock-eye-tracking-thread-and-vtcm)). Once the eye client keeps running, any job 2/4 that reaches the firmware's thread 3 while the broker holds a wake corrupts both sides. The adapter's yield releases the session only for requests it sees in its own blocks. The firmware's handler matches Gazenet's layout but sits at the block this adapter calls FeatureNN, so the offsets above need on-device confirmation before the Gazenet adapter can be called safe. Two fixes would remove the hazard: keep pool 1 out of `0x31700000–0x31800000` in adapter mode, or prove that no job 2/4 can reach `0x1522000` during a wake.

Either Gazenet option sets `Artifacts::keep_eye_running`, so opening the broker session does not pause the eye client. Use this only with the adapter: an ordinary custom session still pauses and restores eye tracking. The launcher equivalents are `FRAME_GAZENET` and `FRAME_GAZENET_VECTOR`, each containing an HXFG prefix. Multiple adapter environment variables may be set together, subject to the session's graph and constant-capacity limits.

`--long-wake` does not enable the CPU-setting tripwire in adapter mode, because stock CDSP activity is expected. `hexecutor/tests/test_xrservice_featurenn.cpp` checks the FeatureNN and Gazenet conversion/result ABIs without a device. No Gazenet model lowering, output equivalence, live camera evaluation, or firmware handoff has been validated yet. Controller/gaze tracking quality on real inputs and long worn sessions remain release gates, as does absence of `(2,0,8)`.

## Persistent state

Autoregressive decode workloads (e.g., LFM3-VL-2B / LFM3-1.7B) require resident per-layer state (such as KV caches and conv-state rings) to survive across sequential `run()` invocations on the DSP without exchanging whole state tensors with the host.

- **Resident DDR Lease Allocation:** Persistent state is allocated outside the per-run activation scratch arena in the private DDR lease (`state` region placed after constants). Because it resides in the mapped DDR lease, persistent state survives across inference steps and routine worker wake rotations.
- **Session Lifecycle & Idle Timeout:** For sessions registered with persistent state (`state_bytes > 0`), `hexecutord`'s idle monitor keeps the session open and holds the DDR lease across idle periods (`--idle-close-ms`). If an unrecoverable device error or explicit release occurs, `hexecutord` flags the state as lost (`state_lost_ = true`) and refuses subsequent `run()` jobs with an explicit error until the client reinitializes state via `reset_state()`.
- **Wire Protocol & APIs:**
  - Control frames: Wire types `RESET = 7` and `RESET_DONE = 8` allow clients to request an explicit state wipe.
  - `Client::reset_state()` and `Session::reset_state()` submit `QDS_OP_RESET_STATE` (4) to the DSP resident engine entry, zeroing the persistent DDR state buffer.
  - Per-step scalar and vector inputs (such as sequence position `pos` or attention mask slices) are submitted cheaply within the inline request frame.
  - Prefill and decode graphs registered under the same `Session` share state tensors by matching non-zero name hashes (`reserved[1]`).
  - Monitoring: Server and client statistics expose `state_bytes` per graph.

## Failure handling

- The broker is the only process that restores firmware state. A fatal device error fails every outstanding ticket with the same error and closes the session through the normal restoration path. The rule in [operations](operations.md) still applies: after a failed restoration the broker does not reopen automatically.
- A client that disconnects loses its streams; queued jobs are dropped.
- The broker refuses to open while the boot is marked as an NPU-cleanup fault or the firmware is already at `(2,0,8)`, as `Session` does now.

## Phases and gates

| Phase | Work | Gate |
| --- | --- | --- |
| 1 | `hexecutord` + `BrokerClient` over one tensor `Session`; hands as first client | Hand viewer through the broker matches its direct-`Session` timings; hour-long session with XRService on the CPU setting |
| 2 | FeatureNN graph registered as a second tensor program; per-graph VTCM budgets; measured device costs; admission | Device timing for FeatureNN; hands and FeatureNN interleaved at 60 Hz without missed deadlines |
| 3 | Combined QDRX + tensor engine, unified memory map; depth worker as a client | Depth exact output unchanged; depth + hands + FeatureNN in one wake at target rates |
| 4 | Stock-job adapter for FeatureNN and Gazenet | Controller and gaze tracking quality equal to stock on captured sessions; no `(2,0,8)` across long worn sessions |

## Open questions

- Which job ids besides FeatureNN 8 and Gazenet 2/4 the stock clients send, and at what rates.
- How long one wake can safely be held with XRService off the CDSP.
- Whether depth needs a mid-inference split point to keep tracking latency acceptable.
- Whether the float FeatureNN graph is close enough to the u8 model, or whether a ReLU-N clamp (the final 1×1 saturates at 3.54) or a u8 tensor path is needed.
