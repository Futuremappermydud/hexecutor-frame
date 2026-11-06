# HMX runs on the Frame

Oct 25, 2026, approximately 21:16–21:44 device time. **A new FP16 HMX
32×32 matrix multiply runs inside Valve's existing CDSP firmware.** Signed
random matrices, identity and zero cases match independent int64 CPU
references exactly. No QuRT boot, FastRPC, replacement firmware, DTB or kernel
change was required for the successful path.

This is a standalone kernel proof. The stereo runner still uses HVX; INT8
HMX, full-model HMX inference and concurrent eye inference have not been
validated. Later tuning below validates large-GEMM FP16 throughput.

**Performance follow-up, approximately 21:58–22:16:** the tuned, VTCM-resident
1024×1024 FP16 GEMM reaches **12.03 TFLOPS**, matching the published V75
result. The main limitation was the unconfigured HMX clock, which was running
at 19.2 MHz. The new harness temporarily applies the shipped 979.2 MHz
operating point and restores it afterwards. See
[Performance tuning](#performance-tuning) below; full-model inference remains
separate from this compute benchmark.

## Reproduce

Sources are in `frame/bench/{hmx_bringup.c,hmx_ops.S,run_hmx.py}`. The matching
working directory on the headset is:

```sh
ssh frame
cd ~/frame-dsp/depth-model/dev/frame
make HEX_CC=clang HEX_LD=ld.lld build/hmx_bringup.elf build/hmx_exec.elf
../.venv/bin/python bench/run_hmx.py --stage 11 --report build/hmx-power.json
../.venv/bin/python bench/run_hmx.py --stage 12 --report build/hmx-ones.json
../.venv/bin/python bench/run_hmx.py --stage 13 --seed 2026 --report build/hmx-random.json
../.venv/bin/python bench/run_hmx.py --stage 13 --case identity --report build/hmx-identity.json
../.venv/bin/python bench/run_hmx.py --stage 14 --iterations 100000 --report build/hmx-benchmark.json
```

Use the Frame's Clang 19.1.7 or Qualcomm's compiler. The existing Clang 14
Docker image cannot assemble this probe's privileged `memw_phys` reads.
HMX instruction words were generated and disassembled with Qualcomm Clang
19.0.07 (`-mv75 -mhmx`), then encoded in `hmx_ops.S` so the shipped open
LLVM assembler can build the final sources. The SDK is no longer a build
dependency. `-G0` avoids unsupported GP-relative relocations and dependence
on the resident firmware's global pointer.

Keep `dsp_service` and SteamVR running. Run only one callback client at a time
and keep the headset awake. The existing transport pauses the eye client,
uses the FLOPS wake-up, checks worker exit markers, then restores code/arena
and restarts the eye client to reload its weights. On a stalled worker it
retains the executing regions. Recovery tests during investigation required
reboots; one recovery required the user to power the headset back on.

Stage 10, the failed test without HMX power-on, is deliberately unavailable
from the CLI. The other numbered stages retain the read-only diagnostics
used during bring-up; stages 4/5 address an older layout and return zeros.

## What makes it work

The firmware and callback ABI checks remain those in `frame_transport.py`:
Valve firmware SHA-256
`4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c`,
protocol 9, loader at `0x87606000`, payload at `0x87610000`, and the upper
8 MB DMA-buffer arena. Only worker 0 owns HMX; the other two workers return.

1. Read the physical configuration table correctly. `cfgbase=0x3018`, so
   its address is `0x30180000`. `memw_phys` takes the byte offset in the
   low 11 bits and the remaining address shifted right by 11, rather than
   a conventional pair of low/high 32-bit words. The table reports private
   subsystem base `0x32380000`, VTCM base `0x31000000`, VTCM size 8192 KB,
   and raw INT8/FP16 HMX rate values 64/32. Do not infer absence from
   `COPROC_TYPE=1`; the HMX rate entries are nonzero.
2. Use the **V75 RSCC layout**, recovered from this device's stock QuRT
   image and checked with live reads. The live RSCC block is `0x323c0000`,
   ID `0x00030200`. H2's older register offsets do not suffice here.
3. Temporarily install only the HMX on/off subsequences into empty RSCC
   sequence memory, verify write-back, trigger power-on and wait for the
   completion latch. No whole-core sleep sequence is installed.
4. Set worker 0's `SSR.XE2` bit 26, execute HMX against VTCM, and restore
   the original SSR afterwards. `mxclracc.hf` alone returns even without
   power setup; that was insufficient evidence of working matrix compute.
5. Run the matching power-off subsequence, verify its endpoint, and restore
   the sequence words and trigger registers. Leave instruction/data pages
   to the existing callback transport's cleanup.

Offsets relative to `0x323c0000`:

| Register | Offset |
| --- | ---: |
| RSCC identification | `0x000` |
| Sequencer busy | `0x404` |
| Sequencer PC | `0x408` |
| Completion latch; bit 31 acknowledges/clears it | `0x410` |
| Software trigger | `0x460` |
| Trigger start address | `0x464` |
| Branch-address array, recovered from QuRT | `0x4bc` |
| Sequence memory, recovered from QuRT and exercised | `0x550` |

The small installed program contains the 32-byte power-on sequence at byte
offset 0 and 36-byte power-off sequence at offset 32. Successful endpoints
are PC **32** and **68**. The first version wrote at the older `0x600`
offset: both requests then ended at PC `0xd0`, so its original
`power_cycle_completed=true` interpretation was incorrect. Its report is
annotated; the corrected tests check both endpoint PCs as well as completion.

The source QuRT image SHA-256 is
`1b0d96ff0363e7db34a8890588e1279e3c67fb570ba6b7f0fc75c7b808cded6b`.
Its 76-word child-sequence array is at ELF file offset `0x27e1f0`, VA
`0xf02411f0`. The HMX off/on subsequences start at original byte offsets
4 and `0x28`; the remaining words include core sleep programs and are not
installed. Provenance and raw words are in `build/hmx-qurt-sequence.json`.
QuRT disassembly also identifies sequence copying at `0xf009fa10`, software
triggering at `0xf009fa94`, and completion handling at `0xf009fce8`.

Public [Qualcomm H2 power code](https://github.com/qualcomm/hexagon-hypervisor/blob/master/kernel/power/hmx/hmx.ref.c)
provided the initial approach; its older address constants were replaced by
the device-specific layout above. The FP16 operations were cross-checked
against [llama.cpp's HMX helpers](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-hexagon/htp/hmx-utils.h).

## Tile layout and validation

For each 32×32 FP16 operand, interleave pairs of rows:
`packed[r//2, c, r%2] = matrix[r,c]`. Use the same arrangement for A[M,K],
B[K,N] and C[M,N]. A, B, C and output parameters occupy VTCM addresses
`0x31000000`, `0x31001000`, `0x31002000`, `0x31003000`.

Each output column's parameter word is `0x00003c00`: FP16 scale 1 and bias
0. Using `0x3c003c00` supplied bias 1 and produced 33 for the all-ones
product; correcting it produced 32 in all 1,024 lanes.

The kernel performs `mxclracc.hf`, `bias = mxmem2`, paired activation/weight
loads, accumulator conversion and a VTCM store. A following HVX VTCM read
forces store completion before collecting results. Random inputs are small
signed integers represented exactly in FP16; their int64 CPU products are
also exactly representable in these tests. This does not establish behavior
for arbitrary FP16 rounding or NaNs.

Verified cases, each checking all 1,024 outputs:

| Case | Result |
| --- | --- |
| All ones, zero output bias | Every output equals 32 |
| Signed random, seeds 1234, 42 and 2026 | Exact against independent int64 matmul |
| Identity right operand | Exact |
| Zero right operand | Exact |
| 10,000 and 100,000 repeated multiplies | Completed; final tile exact |

The final 100,000-iteration loop measured **128,311,288 upcycles**, or
**1,283.11 cycles per tile**. It includes per-tile clearing, parameter loads,
operand loads, conversion/store, function/loop overhead and an HVX completion
read. It excludes host dispatch, packing, power transitions and DDR copies.
That first test did not calibrate the DSP/HMX clock relationship. The later
tuning below establishes the clocks and a comparable large-GEMM measurement.

Final bring-up payload ELF SHA-256:
`7efacd7f6771c5e590c19b3241baa70d37a2b862b07f4bf0f7c9e39e4dd384a0`.
Reports live in `build/hmx*.json` locally and on the device. The final checks
found control `[1,0,1]`, callback 0, empty loader/payload/arena, active
`dsp_service` and SteamVR, and the restored stock eye client running.

Next: implement and validate integer HMX tiles for the recovered quantized
graph, then batch work with dedicated buffers and explicit gaze scheduling.
The existing persistent-convolution protocol bugs in `REALTIME.md` remain
separate from this one-shot HMX probe.

## Performance tuning

The new `bench/hmx_tune.c`, `hmx_tune.S`, `hmx_tune_encoded.S`,
`hmx_power.h` and `run_hmx_tune.py` implement a complete tiled GEMM. The
native assembly is the editable instruction source. The encoded assembly
has identical machine instructions and builds with the Frame's installed
open LLVM; Qualcomm's SDK is only needed to change/reassemble native HMX
instructions. `encode_hmx.py` rejects unresolved text relocations, preserves
exported labels and records source/object hashes.

To regenerate the encoded assembly after changing native instructions:

```sh
hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b \
  -c bench/hmx_tune.S -o build/hmx_tune_native.o
python3 bench/encode_hmx.py build/hmx_tune_native.o \
  --source bench/hmx_tune.S --output bench/hmx_tune_encoded.S
```

Reproduce on the Frame, in the same development directory as above:

```sh
make HEX_CC=clang HEX_LD=ld.lld build/hmx_tune.elf
../.venv/bin/python bench/run_hmx_tune.py \
  --sizes 1024,1024,1024 --flags 4 --reps 1000 --hmx-mhz 979.2 \
  --seed 1234 --report build/hmx-performance.json
```

The baseline uses `--hmx-mhz 0`. The harness supports only the tested subset
of shipped operating points: 364.8, 499.2, 672, 806.4 and 979.2 MHz. It
requires the expected idle clock configuration before changing it. It uses
the running kernel's existing rail votes; it adds no voltage, kernel, DTB,
firmware or persistent clock configuration.

### Findings

The DSP scalar core was already running at **1843.2 MHz**. However, the
dedicated MXU/HMX PLL was disabled (`MODE=0x100`, `L=0`, `OPMODE=0`), and
the HMX RCG selected XO (`CFG=0`): **19.2 MHz**. Merely powering the matrix
engine did not configure its clock. The measured full-GEMM baseline of
236 GFLOPS agrees with that low clock.

The shipped QuRT BSP identifies:

| Clock item | Address |
| --- | --- |
| MXU/HMX Lucid OLE PLL | `0x32341000` |
| Q6/MXU clock-controller base | `0x32348000` |
| HMX RCG CMD / CFG | `0x32348180` / `0x32348184` |
| HMX core branch | `0x323481a0` |

Its HMX operating-point table contains **192, 364.8, 499.2, 672, 806.4 and
979.2 MHz**; the maximum tested value is a shipped operating point. The
table is at QuRT ELF file offset `0x27ab78`, referenced by clock domain
VA `0xf023d908`; its RCG offset is `0x180`. The named clock node is
`turing_q6_cc_mxu_cc_core_clk`, at file offset `0x27a804`. Full provenance
and raw records are in `build/hmx-clock-provenance.json`.

The clock setup follows the running kernel's Lucid OLE PLL startup sequence,
with bounded lock/update waits, then selects the PLL via RCG source 2. The
reference clock is 19.2 MHz, so the 979.2 MHz point uses multiplier 51.
After each command it returns the mux to XO, restores the PLL's modified
writable registers, powers HMX off and restores sequencer memory. Reports
check those restored registers and both power-sequence endpoints.

### Kernel changes

The matrices are packed as 32×32 FP16 tiles. A uses tile order [M,K], B
uses [N,K], and C uses [M,N]. Their 6 MB total fits in the existing 8 MB
VTCM mapping, with additional output-parameter and completion space.
Worker 0 waits until workers 1/2 have completed the stock benchmark before
overwriting the full VTCM region.

The inner kernel loads up to 32 K tiles with `activation.hf ... :deep`
and paired weight loads. It converts/stores once per output tile rather
than once per 32-term contribution. Assembly hardware loops eliminate
stack traffic and avoid synchronization between output tiles. The
accumulator-transfer mode clears it for the next tile. A fused `:after.hf`
store was also tested; it performed similarly.

Moving the loops into assembly did not materially improve the XO-clock
baseline. The major improvement came from configuring the HMX clock.

Timing ends only after an ordered HMX zero store to a separate, freshly
initialized completion tile and its HVX read. This avoids relying on an
output address reused by earlier benchmark repetitions. The completion
word is checked, and all output elements in the final matrix are compared
against an independent int64 CPU GEMM. Inputs are random -1/0/1 values
represented in FP16, making the tested sums exactly representable.

### Results

All rows below use 1024×1024×1024 FP16 GEMM with A, B and C in VTCM.

| HMX clock | Measured TFLOPS | Repetitions | Validation |
| --- | ---: | ---: | --- |
| XO, 19.2 MHz | 0.2362 | 10 | Exact |
| 364.8 MHz | 4.4800 | 10 | Exact |
| 499.2 MHz | 6.1328 | 50 | Exact |
| 806.4 MHz | 9.9040 | 50 | Exact |
| 979.2 MHz, fresh completion marker | 12.0321 | 1,000 | Exact |
| 979.2 MHz, fresh completion marker | **12.0324** | **10,000** | **Exact** |
| 979.2 MHz, final open-LLVM build | 12.0320 | 1,000 | Exact |

The 10,000-GEMM run took **1.784755 seconds**, or **178.48 microseconds per
GEMM**, and is **50.95× faster** than the matched-shape baseline. Its
host-observed invocation took 1.818242 seconds, including dispatch and
tensor copies. Different seeds were checked on repeated runs. Each final
1024×1024 result checks all **1,048,576 outputs**, with zero error; individual
intermediate repetitions are not copied back or separately validated.

[The published V75 benchmark, Table 2](https://arxiv.org/html/2509.23324v1#S3.T2)
reports **12.03254 TFLOPS** for the same GEMM dimensions with all operands
in TCM. The tuned Frame result matches within measurement resolution.

Timing uses the 19.2 MHz architectural timer (`c31:30`); the Frame kernel
reports `arch_timer: cp15 timer running at 19.20MHz`. Independent upcycle
counts give a 1843.2 MHz DSP core clock. `build/hmx-performance-summary.json`
and `build/hmx-tune-final-*.json` contain the measurements and output hashes.
The final tuning ELF SHA-256 is
`f502c05da15ebc8da071b0d73c363cd394232763afa28d400e4f7ef2cb4eb50d`.

These are compute-kernel results, excluding packing, DDR transfers, power
transitions and host dispatch. They do not establish full-model throughput,
long thermal steady state, arbitrary FP16 rounding behavior or simultaneous
gaze inference. The stereo model remains on its earlier HVX path.
