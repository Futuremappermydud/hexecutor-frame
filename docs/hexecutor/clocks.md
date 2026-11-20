# AXI, HMX, and memory bandwidth

Clock state was a major bring-up issue, but it is not a universal explanation for later performance gaps. Distinguish the scalar core, AXI master, and HMX clock domains, and read the registers during the workload being measured.

## AXI-master finding

Early firmware sessions left the Q6 AXI master on the 19.2 MHz crystal while its supplying core PLLs were off. Sequential DDR traffic sustained about 1.25 GB/s. HVX loads, a user-DMA engine, and combinations of both met essentially the same sustained limit. Two DMA engines briefly burst higher, then returned to it.

Raising interconnect votes, a compute-NoC vote, or the GCC NSP-NoC clock did not remove that bottleneck. Configuring the AXI master's shipped 750 MHz operating point did. In a controlled short-context Qwen comparison with identical generated tokens, decode rose from 3.58 to 53.1 tokens/s and DSP cycles/token fell from 516 M to 35 M. This is a memory-streaming workload result, not a depth-inference speedup claim.

`qd_axim.h` records the sequence:

| Item | Address or setting |
| --- | --- |
| AXI clock control | `0x32008254` |
| Supplying core PLL | `0x32002000` |
| Nominal 750 MHz sequence | PLL L = 78, alpha = `0x2000` |
| Optional 1000 MHz definition | L = 104, alpha = `0x2aaa`; requires the existing nominal rail votes |

Only the designated master changes this state. The helper saves the original mux/PLL values in memory that survives wake/packet activity, checks expected initial state, uses bounded lock/update waits, and restores before exit. A failed mux restoration must not disable a PLL still in use. The 1000 MHz definition is not evidence that every workload or firmware state can safely use it.

## Later depth measurements

Live-register probes inside the optimized depth run found AXI near 749 MHz already, with or without `QDS_AXIM_RUN`. HMX was also correctly configured. Raising AXI again consequently gave no gain. One recorded probe measured about 11 GB/s from the uncached backup region and 84 GB/s of one-thread VTCM reads; those access patterns are not interchangeable with arbitrary model traffic.

The durable rule is to verify the selected source/divider and effective bandwidth before tuning votes or adding another DMA engine. A flag being enabled does not prove it changed the hardware, and the early XO result must not be repeated as the state of the later prepared engine.

## HMX clock finding

The scalar core had been measured near 1843.2 MHz, while HMX's own PLL was disabled and its mux selected 19.2 MHz XO. Merely powering the matrix engine was insufficient. The shipped HMX operating points include 364.8, 499.2, 672, 806.4, and 979.2 MHz. The documented maximum tested point uses a multiplier of 51 from the 19.2 MHz reference.

| HMX setting | Recorded large-GEMM throughput |
| --- | ---: |
| 19.2 MHz XO | 0.236 TFLOPS |
| 499.2 MHz | 6.13 TFLOPS |
| 806.4 MHz | 9.90 TFLOPS |
| 979.2 MHz | 12.03 TFLOPS |

These repeated, numerically checked GEMM measurements establish the clock effect. They do not establish model throughput: staging, conversion, metadata, and synchronization are outside peak matrix arithmetic.

The HMX PLL is at `0x32341000`; its clock-controller base is `0x32348000`. Startup follows bounded PLL/mux sequencing and restores prior state. Architectural timer measurements use 19.2 MHz; core upcycles and timer ticks must not be interpreted as the same unit. Later live depth probes confirmed the multiplier 51, rather than inferring it only from a build flag.

Source: [AXI sequence](../../hexecutor/native/qd_axim.h), [HMX power/clock handling](../../hexecutor/native/qd_hmx_power.h), [AXI probe](../../hexecutor/benchmarks/axim_tune.c), and [live depth probe](../../samples/depth-model/deployment/lanes/device_clkprobe.py).
