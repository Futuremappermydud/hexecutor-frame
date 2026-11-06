# CI64 mode object and staging audit

## Mode object evidence

The real generic-leaf caller passes `r5 = r16`, where the outer wrapper provides `r16 = this + 0x20`. The inner HMX setup writes fields beginning at `+0x08`, but does not initialize the first two words. The generic leaf reads word 0 as an enable flag and word 1 as the accumulator selector when enabled.

The QAIRT source headers available locally define `Interface::qparms` as `{offset, scale, scale_recip}` and `ScaleOffsetInterface::get_offset()` as the quantization offset. Recovered QDRX graph tensor 1001 (the output of sequence 1022) is uint8 with zero point 0 and scale `0.011042261496186256`. This is consistent with selector word 1 being the output zero point, and with the fixture’s second word `0`; however, the provider’s `this+0x20` type and its constructor/write path are not exposed in the installed SDK headers. Treat the quantization interpretation as a hypothesis until the writer is traced.

The local include tree contains no `hmx_params` declaration or `ConvLayer` execution implementation. The stripped provider has no debug type information for the internal caller objects. The exact blocker for asserting the mode word is therefore the absent writer/type metadata for `this+0x20`, not a missing graph zero point.

## CI64 fixture memory audit

For `qnn_hmx_generic_xwb_64x64_sim.c`, the staged regions are non-overlapping and fit within the 512 KiB translated VTCM range:

| Region | Range relative to VTCM base | Size |
|---|---:|---:|
| Input pages | `0x00000–0x35fff` | 221,184 B (108 pages) |
| Input guard | `0x36000–0x39fff` | 16,384 B |
| Output pages | `0x3a000–0x49fff` | 65,536 B (32 pages) |
| Output guard | `0x4a000–0x4a7ff` | 2,048 B |
| Weights | `0x4c000–0x54fff` | 36,864 B |
| FI | `0x56000–0x563ff` | 512 B |
| Finish word | `0x57000–0x57003` | 4 B |
| Input table | `0x58000–0x583ff` | 1,024 B |
| Output table | `0x58400–0x587ff` | 1,024 B |
| Completion flag | `0x58800–0x58803` | 4 B |

The simulator log `xwb-ci64-co64-direct.log` reports early outputs saturated to 255 and 65,415 mismatches. The fixture checks input guard, output guard, and unused input/output table entries before printing the final output-oracle failure; those checks passed. This excludes the prior class of staging-buffer overrun/overlap bug for this fixture, but does not validate CI64 native weight packing or descriptor semantics.

## Current conclusion

CI64 remains unproven. The direct CI64/CO64 probe is a real bounded failed fixture rather than an all-zero result; the CI64/CO32 raw and hypothesized main/tail variants also fail. The exact remaining kernel unknowns are (1) writer/type semantics for `this+0x20` and its second word, and (2) CI64/CO32/CO64 spatial weight packing and depth-phase configuration. Do not mark CI64 safe or unsupported from these probes.
