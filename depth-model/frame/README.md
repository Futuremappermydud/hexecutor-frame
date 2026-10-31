# Model stereo model on the Frame callback

Native runtime work now runs 823 convolutions through original HMX leaves.
Experimental resident warm inference is **647–652 ms**, with inexact arithmetic;
the frozen exact resident baseline is ~861–867 ms and the existing exact hybrid
is ~115 ms. Model's original program is ~9.7 ms. The prepared native executor now has Frame validation for bounded 1x1, input
padding, ADD and LinearClip ranges using resident VTCM tensors and L2 metadata.
Full memory-resident inference near20ms remains unproven. Prepared 3x3 now
passes bounded Frame checks with input-only staging; its zero-copy scattered
input path still has an unresolved device fault. See
[the current handoff](HANDOFF_20261030_2145.md) for flags, reports, and limitations.

> **Oct 29, 2026:** the [optimized hybrid ARM/HMX port](DIRECT.md) runs
> primary smoke inference in 0.401–0.403 s warm (0.530 s first inference),
> with exact simulator outputs. `dist/depth-model-direct.zip` contains the
> updated source and payload; timings below describe historical builds.

> **Oct 25, 2026:** the source tree has moved ahead of the packaged build:
> a loader + persistent VTCM conv payload, with known bugs. See
> [REALTIME.md](REALTIME.md) for the profile, findings and status. The
> instructions below describe the validated `dist/` package.
>
> **Later follow-up:** standalone 32×32 FP16 **HMX** matrix multiplication
> now runs on the Frame and matches CPU references. See [HMX.md](HMX.md)
> for the power setup, source and validation. The stereo runner below still
> uses HVX; it has not been migrated to HMX.
> Performance tuning subsequently reached **12.03 TFLOPS** on a 1024³
> VTCM-resident FP16 GEMM, matching the published V75 result; see HMX.md.

This package runs the complete recovered primary or alternate graph, with all
1,036 convolution tiles lowered to 128-byte HVX dot products. Python/NumPy on
the ARM CPU handles the graph, layout transforms, nonconvolution kernels,
runtime weights and exact integer requantization. It does not require QNN,
QuRT, FastRPC or HMX. This is a hybrid backend, not a wholly DSP-resident graph.

## Build and package locally

```sh
docker build --platform linux/amd64 -t depth-model-hvx-build depth-model/frame
docker run --rm --platform linux/amd64 -v "$PWD/depth-model/frame":/src \
  depth-model-hvx-build make all
python3 depth-model/frame/build_package.py
```

`frame/dist/depth-model-frame.zip` contains both models, the Hexagon ELF, the
runner, callback sources and checksummed execution artifacts. No simulator
memory snapshots or original QNN runtime libraries are included.

## Run on the Frame

Extract the archive, enter `depth-model-frame`, and install NumPy in a venv:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python frame/probe_hvx.py --report hvx-probe.json
.venv/bin/python run.py --smoke --output results/primary
.venv/bin/python run.py --model alternate --smoke --output results/alternate
```

Smoke checks both output hashes automatically. For arbitrary native inputs:

```sh
.venv/bin/python run.py --model primary --inputs INPUT_DIR --output OUTPUT_DIR
```

`INPUT_DIR` contains `x.raw`, `y.raw` (81,920 uint8 bytes each), `hidden.raw`
(163,840 float32 values), and `t.raw` (32 float32 values), with the recovered
native layout. `--compare REFERENCE_DIR` checks every output byte.
`--verify-matmuls` additionally checks each returned DSP dot product against
int64 CPU arithmetic. Outputs and an explicit backend/timing report are saved.

The transport uses the existing stock FLOPS command to wake the DSP workers.
**Eye inference is paused for the session and restored afterwards.** Keep
`dsp_service` and SteamVR running. Run one callback client at a time. This is
not a tested concurrent gaze/inference scheduler or a real-time camera app.
Keep the headset awake during a run: system suspend reboots the DSP. This SSH
session could not acquire a sleep-blocking logind inhibitor without administrator
authentication. The runner detects a lost callback and avoids restoring an old
session over a freshly rebooted DSP.

Before installing anything it checks the exact firmware/protocol, idle state,
callback ownership and unused code/data regions. It holds the eye client with SIGSTOP so SteamVR cannot immediately respawn
a competing client. It restores its code page, work control and arena after
workers finish, then terminates/resumes the paused client so a fresh process
reloads its gaze weights. It allows SteamVR to restart that client first, with
the saved arguments and affinity as the fallback. On a worker timeout it retains
the callback/arena instead of overwriting executing code or restarting gaze.

## Layout and arithmetic

The callback is a freestanding Hexagon v68-compatible ELF with HVX128 enabled;
it uses instructions available on the Frame's V75 CDSP. Its entry is
`0x87605000`, in a dedicated 4 KB code page after the earlier test pages.
The command/data arena is DMA-buffer offset `0x1800000`, length 8 MB (the upper 8 MB of the shared buffer). Its
cached DSP view is `0x88800000`; control/status use the uncached `0xf1800000` view.
The arena must be empty before use. The code page must be empty or contain
the identical previously registered callback binary. The firmware hash is
`4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c`.

Every different callback binary needs a fresh instruction-cache page or
explicit I-cache invalidation. A persistent per-page hash record rejects
replacing this payload at the same address with a different binary. Changing
the code address requires updating the linker and matching protocol constants.

The matrix packet contains signed int16 A[M,K], signed int32 B[K,N] holding
int8-range weights, and int32 C[ceil(K/256),M,N]. N is padded to 32. Three hardware
workers own interleaved output rows. u16 activations are centered around 32768;
the host adds the exact correction in int64. Each 256-term partial is bounded
by 2**30, and the host combines partials in int64 before the calibrated Fi
requantization. DSP inputs are invalidated and outputs cleaned explicitly.

## Local validation

```sh
make -C depth-model/frame build/libqd_host.so
python3 depth-model/frame/test_local.py --report depth-model/frame/build/local-tests.json
python3 depth-model/frame/run_frame.py --backend native-test --smoke --output /tmp/qd-native
```

`native-test` compiles the production command dispatcher with a scalar vector
substitute; it tests packing, indexing, partial sums and bounds without claiming
HVX execution. `packed-cpu` tests the same packets using NumPy; `numpy` selects
the earlier reference. The default backend is `hvx`, with no CPU fallback.

Local checks cover boundary/random dot products, u8/u16 centering and malformed
commands. Full-model smoke and seed-1234 random outputs match for both models
through the native dispatcher substitute. Hardware checks on the Frame also passed:

| Case | Native outputs vs simulator | Seconds |
| --- | --- | ---: |
| Primary smoke, plus CPU verification of every dot product | Both exact | 191.3 |
| Primary seed-1234 random input | Both exact | 136.7 |
| Alternate smoke | Both exact | 141.2 |
| Alternate seed-1234 random input | Both exact | 137.4 |

Every run executed 1,036 HVX convolution calls (26,834,632,704 integer MACs).
Normal inference is roughly **2 minutes 20 seconds**, not real time. About
104 seconds was measured in DSP dispatch/wait; the rest includes ARM graph
operations, packing, memory copies and eye-client restoration. Compact address
maps use about 1.77 GB; other working memory is additional.

The first three runs used code page `0x87604000` and arena offset `0xc00000`.
After a suspend interrupted another test, the published build moved to a fresh
code page and the upper arena described above. Its direct hardware probes and
full alternate-random run were revalidated. The arithmetic is unchanged; each
report identifies the callback binary by SHA-256. The old payload was removed
only after verifying its exact bytes and idle state.

The published callback also passed three signed/boundary/chunking hardware
probes, including sums larger than int32, and cancellation during an active
request. The post-run check found callback pointer 0, restored code/arena,
normal operations state `[1,0,1]`, an active `dsp_service`, and a running stock
eye client. Reports are in `frame/build/*.json` in the source tree and
`frame/validation/` inside the archive.

These results validate the **hybrid ARM + HVX backend** against the x86
simulator. They do not establish QNN loading, HMX execution, original V69
hardware equivalence, real-time performance, or calibrated Frame camera depth.
