# Model depth on Frame

The port runs both recovered Model stereo graphs using exact integer HMX matrix
convolution, with HVX fallback. Native C accelerates ARM tensor access, packing,
collection and requantization. HMX results receive their final exact u8
quantization on the DSP, halving output transfer size; native ARM add/requant
operations avoid NumPy int64 temporaries. Immutable HMX suitability decisions
are cached, and eye-client checks reuse a proc file descriptor. Python still schedules the graph. This is a hybrid
ARM/Hexagon implementation; it does not execute the original QNN context.

## Performance and validation (Oct 29, 2026)

Primary smoke inference with HMX takes **0.401–0.403 seconds warm**, about
**2.5 frames/second**, and 0.530 seconds on the first inference. The earlier
Oct 25 build took 137–141 seconds per inference. These are whole-graph timings;
startup, loading and cleanup are excluded from the per-inference numbers.
`runs[i].seconds` records inference time; top-level `seconds` includes the session.
This is substantial progress, but remains below real time.

Each inference has 1,036 convolution tiles, 26.83 GMAC and 1,146 DSP requests.
916 convolutions use integer HMX; other cases use exact HVX convolution.
The address cache is now about 2.6 MB, down from 1.77 GB. The host uses CPU 7
by default. The payload raises the AXI-master clock to 750 MHz and HMX to
979.2 MHz, and restores prior hardware state on exit.

The HVX version passed all 25 simulator probes plus a randomized-input case
for each of both models. The HMX version is checked with the same full-model
suite and independent int64 convolution tests. Reports ship in `validation/`.
Both output heads must match the reference SHA-256 values exactly. Native ARM
add/requant tests exhaust all u8 input pairs across 31 recovered parameter sets
(over two million comparisons). The HMX simulator also checks five convolution
shapes, including a depth requiring multiple matrix accumulation chunks.

## Run

Unpack the archive on Frame, then use the existing environment:

```sh
cd ~/frame-dsp/depth-model/depth-model-direct-v5
~/frame-dsp/depth-model/.venv/bin/python run.py \
  --left examples/left.npy --right examples/right.npy --output results/pair
```

For another Python environment, install `requirements.txt` and have a C compiler
available (`cc`). The small native ARM helpers compile on first use.
The image runner uses HMX by default. `--compute-engine hvx` selects the vector
path. `--repeat 3` repeats in a single session; `--cpu -1` disables host affinity.

Images must be 256 pixels wide by 320 high in rectified model coordinates.
PNG/JPEG are converted to grayscale; uint8 `.npy` `[320,256]` and native `.raw`
are accepted. `--resize` explicitly enables resizing. It does not rectify images.
`--model alternate` selects the other recovered graph.

Optional `--hidden state.npy` supplies float32 `[1,2,320,256]`; `--transforms
matrices.npy` supplies float32 `[1,2,4,4]`. Defaults are zero hidden state and two
identity matrices, experimental single-pair placeholders. Outputs are not fed
back as state automatically.

The runner saves `outputs.npz`, separate arrays, two PGM previews, `report.json`
and detailed `hardware-report.json`. `float_output` preserves both float32
planes `[1,2,320,256]`, which track horizontal disparity in synthetic probes.
`auxiliary_u8` preserves the native head; `auxiliary_dequantized` applies its
recovered quantization. Camera geometry, temporal-state semantics and metric
calibration remain unresolved. The tool does not produce calibrated metric depth.

To check native smoke inputs against simulator hashes:

```sh
python frame/run_frame.py --smoke --compute-engine hmx --repeat 3 \
  --output results/smoke
```

Use `--inputs INPUT_DIR` for native `x.raw`, `y.raw`, `hidden.raw`, `t.raw` inputs.
`--reference-hashes REPORT.json` verifies saved emulator hashes and `--compare
DIR` compares raw outputs. `--no-clock` leaves the AXI setting unchanged.

## Session and build

Use one DSP callback client at a time, with SteamVR and `dsp_service` running
and the headset awake. The transport checks firmware identity, idle controls,
callback ownership and empty payload/arena regions. It pauses eye tracking
for the session and restores the eye client, callback, payload and arena on exit.
Successful tests restored controls `[1,0,1]`, zero callback and hardware clocks.

Keep the validated loader ELF unchanged: its SHA-256 is
`828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572`.
Code-page history requires this exact identity. Rebuild only the payload:

```sh
make -C frame HEX_CC=clang HEX_LD=ld.lld build/qd_conv.elf
python frame/tests/test_conv_publication.py
python frame/tests/test_quantized_conv.py --hardware --hmx
python frame/bench/verify_optimized.py --hmx --report frame/build/depth-hmx-validation.json
python frame/build_direct_package.py
```

Protocol v6 publishes commands with an ARM memory fence and sequence-tagged,
bounded worker barriers. HMX operands use exact wide integer conversion and
bounded depth chunks. HVX scatter preparation drains before HMX consumption.
`depth-package.json` lists checksums for all packaged files, verified before
running. The chat installation remains separate.

## Resident executor development

`resident_program.py` provides the audited offline IR; `resident_image.py`
lowers both graphs into compact QDRX images (1,738,088 / 1,736,920 bytes).
The freestanding `qd_resident.c` reader and serial dispatcher core passed native
integrity/control-flow tests and 28 decoder-only hardware cases across two Frame
sessions. These probes execute no model kernels. Valid-image checks currently
take 3.3846–3.3888 seconds through uncached memory, including upload/completion;
this is setup validation, not inference latency.

The separate development executor `native_graph.py` now runs all graph kernels
in C on ARM and dispatches convolution to the DSP. Both models pass all 27
smoke/probe/random cases, plus 20 sustained warm runs each. With CPU 7 and
helper cores 6,5,4, current default warm medians are **122.5 ms primary / 122.2 ms
alternate** (about 8.2 inferences/s), with first inference 168.7/164.3 ms.
These exclude graph creation and transport setup. It remains a hybrid ARM/DSP
development path; v5's packaged runner and the performance numbers at the top
of this document are unchanged.

An experimental `NativeGraph.set_block_output(True)` / verifier
`--block-output` mode lets DSP quantization emit croutons directly into the
arena and copies whole 2 KB blocks to graph pools. It passed the same 54 cases
and 40 sustained runs, but medians of 121.8/121.2 ms are only a small improvement
with overlapping timing ranges. It is **off by default**, requires the current
development payload and has no feature negotiation with older payloads. It
does not establish full DSP residency or the anticipated 10–15 ms saving.
The revised payload also passed all 52 probe/random checks on the older
Python/HMX execution path.

A safe full-graph DSP pool layout, DSP glue kernels, persistent resident graph
workers, point-cloud integration and a v6 package remain outstanding. The native
path still returns intermediate convolution results to ARM. See
[NATIVE_EXECUTOR_HANDOFF.md](NATIVE_EXECUTOR_HANDOFF.md) for current sources,
commands, reports and measurements, [RESIDENT_FORMAT.md](RESIDENT_FORMAT.md) for
the wire format, and [DSP_RESIDENT_HANDOFF.md](DSP_RESIDENT_HANDOFF.md) for the
original acceptance gates.

Later Oct30 development update: the opt-in native integer HMX payload reaches
**99.6 ms primary /100.4 ms alternate** median over all54 cases with direct
block output, versus114.5/114.9 ms exact in the same experiment. Native modes
change rounding and final outputs; the exact backend remains the default.
Synthetic stress cases amplify the difference up to98 u8 levels, so this is a
performance experiment, not depth-quality acceptance or a packaged upgrade.

A full DSP-resident prototype now exists, but a direct port over shared DDR is
much slower (~1.7 seconds warm). Corrected memory/cache handling passes the
previously failing cold-random cases. Native kernels that consume VTCM croutons
directly are the next route being tested; the original aligned1x1 leaf passes
128 ISA fixtures. See `RESIDENT_EXECUTOR_WIP.md` for current acceptance status
and `NATIVE_EXECUTOR_HANDOFF.md` for complete measurements. Installed v5 and
the point-cloud application remain unchanged.
