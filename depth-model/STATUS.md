# Model depth project index

Status snapshot: **2026-11-01**. The newest consolidated Frame runtime checkpoint
is the [Oct 30 evening master handoff](frame/HANDOFF_20261030_2145.md); its
bounded-call results and limitations are supplemented by the dated validation
reports linked below. Treat older handoffs as historical checkpoints unless
the master handoff points to them for a specific lane.

## Start here

- [Model recovery, simulator behavior, and full-graph reference](README.md) —
  extraction, tensor contract, arithmetic, and durable model evidence.
- [Frame callback overview and package instructions](frame/README.md) — the
  exact hybrid path and how to build/run the packaged callback.
- [Frame direct backend](frame/DIRECT.md) — current hybrid implementation and
  measured results.
- [Frame master handoff](frame/HANDOFF_20261030_2145.md) — current experimental
  runtime checkpoint, device state at handoff, validated bounded ranges, and
  restart order. It is the navigation point for the prepared, stride-2, and
  CH/u16 lane handoffs.
- [Resident executor work log](frame/RESIDENT_EXECUTOR_WIP.md) — detailed
  chronological evidence for the experimental resident path.

## Runtime and evidence lanes

| Lane | Current reading order | Status boundary |
| --- | --- | --- |
| Extracted model and QNN reference | [Project README](README.md), [reference README](reference/README.md) | Both original contexts run in the QAIRT 2.50 x86 HTP simulator; recovered arithmetic/reference evidence is separate from Frame runtime claims. |
| Frame exact hybrid | [Frame overview](frame/README.md), [direct backend](frame/DIRECT.md) | Exact simulator outputs are documented for the hybrid path; use the linked reports for hardware timing and tested artifact details. |
| Experimental resident graph | [Master handoff](frame/HANDOFF_20261030_2145.md), [resident work log](frame/RESIDENT_EXECUTOR_WIP.md) | Latest consolidated handoff reports 647–652 ms warm and inexact. This is not the exact hybrid baseline and does not establish the 20 ms target. |
| Prepared native program | [Format](frame/native_program/FORMAT.md), [C API](frame/native_program/C_API.md), [executor notes](frame/native_program/EXECUTOR.md), [prepared lane handoff](frame/native_program/HANDOFF_20261030_2145.md) | Explicitly registered handlers cover additional bounded command families; host/V75-simulator CPU-prefix checks and Frame validations each prove only their named ranges. The new [reference kernel](frame/native_program/prepared_reference.h) runs every command of both models through the executor on the host (see the [reference-executor handoff](HANDOFF_20261101_REFERENCE_EXECUTOR.md)) and, as the frozen reference engine, on Frame hardware (exact, ~15.5 s; see the [full-run handoff](HANDOFF_20261101_FRAME_FULL_RUN.md)). See the [mixed-range report](frame/PREPARED_MIXED_RANGE_FRAME_VALIDATION_20261030.md), [Quantize report](frame/PREPARED_QUANTIZE_FRAME_VALIDATION_20261030.md), and [Oct 31 SlicePad CPU-prefix simulator report](frame/native_program/SLICE_PAD_CPU_PREFIX_SIM_REPORT_20261031.md). |
| Stride-2 and CH/u16 investigations | [S2 lane handoff](frame/native_runtime/stride2/HANDOFF_20261030_2145.md), [S2 ABI trace](frame/native_runtime/stride2/v73_capture/ABI_TRACE.md), [CH/u16 lane handoff](frame/native_u16/HANDOFF_20261030_2145.md), [CH basis follow-up](frame/native_u16/ch5099_crossed_basis_report.md) | Diagnostic/provider evidence is not an integrated exact native kernel. Respect each report's stated proof boundary. |
| Historical resident/runtime checkpoints | [Oct 30 PM handoff](frame/HANDOFF_20261030_PM.md), [native executor handoff](frame/NATIVE_EXECUTOR_HANDOFF.md), [resident DSP plan](frame/DSP_RESIDENT_HANDOFF.md) | Keep for experiment history and design context. Use the master handoff for consolidated Frame status. |

The prepared lane has bounded coverage for blocked ADD/REQUANT/LinearClip,
BlockZap, spill/fill, flat-to-VTCM, space-to-depth, InputSlicePad, Abs, Mean,
stride-2 convolution, and CH/u16 convolution. The Oct 31
[SlicePad report](frame/native_program/SLICE_PAD_CPU_PREFIX_SIM_REPORT_20261031.md)
compares commands 68–69 against captured CPU-prefix output in the V75
simulator, not on Frame hardware. The new [Abs](frame/native_program/run_prepared_abs_sim.sh),
[Mean](frame/native_program/run_prepared_mean_sim.sh),
[stride-2](frame/native_program/run_prepared_conv_s2_sim.sh), and
[CH/u16](frame/native_program/run_prepared_conv_u16_sim.sh) runners check
selected captured CPU-prefix commands under the simulator; these are bounded
checks, not Frame validation. In particular, stride-2 coverage is three primary
commands, CH/u16 coverage is two primary commands, Mean covers one command per
model, and Abs covers selected FH commands in both models plus one primary CH
command. BlockZap,
spill/fill, flat-to-VTCM, and space-to-depth details and proof limits are in the
[executor notes](frame/native_program/EXECUTOR.md#additional-bounded-command-handlers).

New isolated prepared registrations (Oct 31) cover these families. Each one
passes a host preflight census over every matching command, then compares
whole physical pools bit-for-bit against captured CPU-prefix fixtures on
the host and in the V75 simulator:

| Handler | Commands accepted per model | Host full-pool fixtures | V75 full-pool fixtures |
| --- | --- | --- | --- |
| [Sign](frame/native_program/prepared_sign.c) var 40 FH + var 39 CH | 13/13 | 26 (all 13 per model) | 4 (1 FH + 1 CH per model) |
| [AddSubFlat](frame/native_program/prepared_addsub_flat.c) var 62 | 3/3 | 6 (all) | 6 (all) |
| [MulFlat](frame/native_program/prepared_mul_flat.c) var 75 | 2/2 | 4 (all) | 4 (all) |
| [Reshape CB](frame/native_program/prepared_reshape_cb.c) var 49 | 80/80 | 16 (source H pad 0–7, 1 command each per model) | 6 (pads 0, 3, 7 per model) |

They are callable only when explicitly registered; they are not yet part of
the Frame runner registration or a validated multi-command prefix. No Frame
hardware result exists for them. CH Mul (10 commands per model) is still
uncovered. See the [executor notes](frame/native_program/EXECUTOR.md#new-direct-scalar-handlers).

The Nov 1 [reference kernel](frame/native_program/EXECUTOR.md#full-model-reference-kernel)
ports every `qd_graph.c` CPU kernel and covers all 76 (opcode, variant) keys.
On the host, both models pass in reference-only mode and in mixed mode (the
18 specialized handlers chained over the reference). In both modes, every
command leaves all seven pools identical to `qd_graph` CPU, and the
whole-schedule run produces the emulator smoke output sha256. The scratch peak
is 1,647,936 bytes. In the V75 simulator, the reference kernel reproduces the
captured CPU prefixes (primary commands 0–131, alternate 0–99) bit for bit.
With the exact HVX fast conv path (all 1036 convs per model), the full schedule
of both models passes in the V75 simulator in 6 segments per model, ending at
the emulator smoke sha256, at about 6.5e9 handler pcycles per model (see the
[reference-executor handoff](HANDOFF_20261101_REFERENCE_EXECUTOR.md)). On
**Frame hardware** the frozen `QDS_PREPARED_REFERENCE=1` engine (`c432a714...`)
runs every command of both models in one prepared call. Smoke and random
outputs equal the emulator sha256, cold and without a pool reset. It takes
about 15.5 s per inference, and the lease restored cleanly in every verifier
session. The OpenXR point-cloud app can select it (`--depth-model-prepared`);
its depth on a captured pair is byte-identical to the v5 hybrid. One app
session's shutdown left `dsp_service` at control `(2,0,8)`, which needed a
reboot; this is unexplained. Results, cycle profile and next steps are in the
[Frame full-run handoff](HANDOFF_20261101_FRAME_FULL_RUN.md).

The [plan-cache prototype](frame/native_program/nprg2_plan_cache.h) and its
[test source](frame/native_program/test_nprg2_plan_cache.c) exercise plan
building and cached dispatch with mock handlers across both prepared command
tables. They are not integrated into the executor/runtime, and the mock test
does not establish numerical command execution. Handler registration remains
explicit; these bounded checks do not establish full prepared-model execution
or end-to-end timing.

## Test and validation entry points

- Model reference: `python3 depth-model/tools/verify_htp_emu.py` and the
  simulator commands in [the project README](README.md#simulator-execution-and-reference-arithmetic).
- Frame package/local checks: commands under “Build and package locally” and
  “Local validation” in [the Frame README](frame/README.md).
- Prepared-program host and simulator checks: see the run commands in the
  [prepared lane handoff](frame/native_program/HANDOFF_20261030_2145.md) and
  individual reports. Hardware validation is bounded to the ranges each report
  names; do not infer full-model acceptance from those checks.

## Handoff maintenance

Handoffs are retained in place to preserve provenance and existing references.
The dated master handoff is the consolidated entry point; lane handoffs remain
the detailed evidence for their own scope. When a newer report supersedes a
checkpoint, link to the report instead of copying its results into multiple
handoffs. No experiment log or unique technical evidence is archived or removed
by this index.
