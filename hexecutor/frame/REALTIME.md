# Toward real-time inference on the Frame (Oct 25, 2026)

> **Oct 29 update:** the four protocol/order defects listed below are fixed in
> the current sources. Both full models now pass on hardware with the AXI clock
> fix: 11.5–11.9 s warm, 23–24 s first inference. See [DIRECT.md](DIRECT.md)
> for current measurements and packaging. The text below is the earlier log.

Status notes from the session that profiled the hybrid backend (see
[README.md](README.md)) and started a faster NPU path. **Work in progress:**
the source tree is ahead of `dist/depth-model-frame.zip`, which is still the
validated slow build. Read [Known bugs](#known-bugs-fix-before-the-next-hardware-run)
before running anything new on the headset.

**Later HMX follow-up:** [HMX.md](HMX.md) records successful standalone FP16
HMX matrix multiplication, exact CPU comparisons and a 100,000-iteration
kernel loop. The convolution/model path described here remains HVX, and
its persistent-protocol bugs still need fixing.

## Where the 2 m 20 s goes

Measured on the Frame with `bench/profile_run.py` (cProfile plus per-call hooks):

| Part | Seconds | Notes |
| --- | ---: | --- |
| Waiting for the DSP | 103.6 | 1,036 calls at a flat ~0.25 GMAC/s, regardless of shape |
| Packing matrices into the arena | 0.3 | |
| Reading results back | 1.3 | |
| Python HTP emulator on the ARM | ~40 | 17 s of this is the one-time address-map build |

On the Mac (`bench/profile_local.py`, native scalar C convs), a full run is
10.7 s. About 7.5 s of that is the convs and ~3 s is the Python emulator. The
whole graph touches only **~337 MB per inference**, so neither cost is
fundamental.

## Why the DSP was slow: Valve's memory setup

Microbenchmarks (`bench/bench_hvx.c`, `bench/run_bench.py`; DSP clock ~1.9 GHz):

| Measurement | Result |
| --- | --- |
| HVX `vmpyi` (16x32-bit), registers only, 3 threads | ~91 GMAC/s |
| HVX `vrmpy` (u8 x s8), registers only, 3 threads | ~366 GMAC/s |
| HVX load from DDR (arena, cached or uncached view) | ~190 cycles per 128 B, no overlap; ~1.3 GB/s total |
| HVX load from VTCM | ~6 cycles |
| Scalar load from L2-TCM | ~5 cycles; from VTCM ~45 cycles |

Valve's boot code sets SYSCFG with L2CFG=0, so **the whole L2 is TCM and there
is no L2 cache**. HVX loads bypass L1, so every HVX load from DDR goes to DRAM.
The old kernel re-streamed int32 weights from DDR for every output row and was
load-bound.

Firmware TLB entries (from its boot code):

| VA | Size | Use | Cache attr |
| --- | --- | --- | --- |
| `0x87000000`, `0x88000000` | 16 MB each | firmware image, DMA buffer (arena at `0x88800000`) | 7 (write-back) |
| `0xf0000000`, `0xf1000000` | 16 MB each | uncached views of the same | 6 |
| `0x30000000` | 1 MB | L2-TCM | 7 |
| `0x31000000`, `0x31400000` | 4 MB each | VTCM (8 MB) | 7 |
| `0x32000000` | 16 MB | DSP subsystem registers | 4 |

The stock FLOPS wake clears VTCM `0x31000000 + tid*0xF0000` (960 KB per
thread) and L2-TCM on every call, so those regions are safe scratch for our
payloads.

96% of the MACs (25.8 of 26.8 GMAC) are u8 activations x int8 weights, which
suits `vrmpy`. Every conv call is an 8-row tile: input ≤ ~170 KB, weights ≤
295 KB. The geometries are listed by `bench/conv_geometry.py`. **HVX floor:
~80-100 ms/frame** at full efficiency, i.e. about 10 Hz. The Model reaches real
time on HMX. Standalone FP16 HMX now works here (see [HMX.md](HMX.md));
the quantized model has not been ported to it.

## What was built

All in this directory. Builds with `make all` in the existing Docker image;
`libqd_conv_host.so` builds on the host.

| File | Purpose |
| --- | --- |
| `qd_loader.S` | Fixed loader page at `0x87606000`. Invalidates D/I-cache over the 32 KB payload region at `0x87610000` and jumps there, so payloads can change without burning code pages. Never change it once a DSP session has run it |
| `qd_payload.ld` | Linker script for payloads at `0x87610000` |
| `qd_conv.c`, `qd_conv.h`, `conv_protocol.py` | Persistent direct-conv payload: copies input to VTCM and weights to L2-TCM, LL/SC barrier, `vrmpy` with 32 pixel lanes x 2 pixel vectors x 8 channels. u16 inputs are split into hi/lo byte planes; zero-point correction stays on the host in int64 |
| `qd_fence.c` | ARM `dmb` helper, compiled on first use with `cc` (needed for the persistent protocol) |
| `frame_transport.py` | Now takes `(loader, payload)`; adds `conv`, `submit`, `stop_persistent`, `workers_exited` and a "stuck" path that keeps code installed if workers don't exit |
| `frame_hvx.py` | `HvxConvolution(direct=True)` uses the conv path; 1x1 convs are flattened to one pixel row |
| `run_frame.py` | Defaults now point at the loader + `qd_conv.elf`; `native-test` uses `libqd_conv_host.so` |
| `test_conv_local.py` | Exact check of the conv dispatcher on every model geometry, with boundary values |
| `bench/` | Microbenchmarks, one-shot probes, profilers and `dsp_state.py` (read-only state check) |

Results:
- **Local:** `test_conv_local.py` passes everywhere, and the full primary model is bit-exact through `--backend native-test` (both outputs match the recorded hashes).
- **Frame:** single convs are exact. A (10,34,128)x(3,3,128,128) tile takes ~0.5 ms of DSP time (copy ~70 µs, compute ~300 µs), about 70x the old kernel. Compute is at ~1/3 of `vrmpy` peak. Host-side overhead (~5 ms per call, packing and polling) now dominates.
- **Not yet done:** a full-model hardware run on the new path.

## Known bugs (fix before the next hardware run)

1. **Barrier counter overlaps the command.** `QDC_BARRIER_OFFSET` (64) is the
   command's `idle_mcycles` word. After each conv the counter (1-3) becomes the
   idle timeout, so workers exit about 1.6 ms later whenever the ARM is slower
   than that. Move the counter outside the 80-byte command, for example to
   offset 96.
2. **Header race.** `pack_conv` and `pack_exit` write the whole header,
   sequence field included, so `sequence` briefly reads 0. Write the header
   without the sequence field, then `dsb`, then the sequence. On the DSP,
   re-read the sequence after loading the header and retry if it changed.
3. **Barrier can hang.** `arrive_and_wait` has no timeout. A worker that
   misses a command leaves the others spinning inside payload code forever.
   Add a cycle-limited wait that reports an error.
4. Use `dsb sy` rather than `dmb sy` in `qd_fence.c`; the DSP is outside the
   ARM's shareability domain.

## Hazards learned the hard way

- **CTRL==3 does not mean the workers are done.** dsp_service's request
  thread sets it and times out after 100 ms ("Timeout reached while waiting
  for the CPU to read the results") no matter what the workers are doing.
  Only the payload's exit markers (`QdStatus.reserved == 'EXIT'`) prove that no
  worker is executing payload code. Never restore code pages without them.
- **Reading DSP subsystem registers from Linux (`/dev/mem` at `0x32300000`)
  resets the whole SoC.** The kernel driver drops those clocks after the DSP
  boots.
- **Some DSP-side register reads stall the worker forever.** Recovery needs a
  reboot.
- After a hand release, CTRL0 can stay at 2. Reboot to get back to `[1, 0, 1]`.
- Recovery: `sudo systemctl reboot` (eye tracking and dsp_service come back
  on their own). Check with `python3 bench/dsp_state.py`.

## Next steps

1. Fix the four bugs above, then run `bench/probe_conv.py` and a full
   `run_frame.py --smoke` on hardware.
2. Tune the conv kernel: software-pipeline loads against `vrmpy`, try more
   accumulators, and cut host overhead (pack in C or NumPy views, tighter
   polling).
3. Replace the Python emulator with a compiled exact replay of the same node
   sequence: precomputed address maps and C kernels. It stays bit-exact
   against the existing reports, and should take tens of milliseconds on the
   ARM.
4. If more than ~10 Hz is needed, move the convs to the GPU (Vulkan integer
   dot product). A logical-graph lift (the 169 source ops: 67 convs, 28 adds,
   18 muls, slices, resizes, permutes, ...) is the longer-term clean
   representation. Tensor-ID edges connect only about half the ops; the rest
   need byte-level provenance tracing through the emulator's pools.
