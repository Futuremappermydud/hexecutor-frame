# Depth Primary Model: Performance and Error Baseline (2026-11-05)

Starting point and follow-up record for review and optimization of the Frame depth engine.

**Latest status (2026-11-05):** see [section 12](#12-exact-planes-and-plan-replay--2026-11-05-current-status). The accurate primary candidate now measures **14.598 ms** (14.284 ms with an optional check removed) and is byte-identical to the same mixed-rounding host oracle on all **27 synthetic cases on Frame** (54 of 54 runs), so its result against the working f32 error gate is unchanged from the 42 ms build. The **~10.5 ms (4× from 42 ms) target is not yet achieved**. Section 11 records the 42 ms state; sections 1–10 are the original baseline, and their native-rounding timings are not the current accurate candidate.

The follow-up includes corrected phase attribution, all rejected experiments, new Model measurements, the official QAIRT 2.25 download and simulator comparison, current build provenance, and the limits of the 4× projections.

**Goal (user, 2026-11-05):** minimize primary depth inference time **without error surpassing Model's**.

**Original baseline status:** the speed numbers in sections 1–10 were all measured with hardware rounding, which **does not meet that error goal**. Frame is about 2.7× Model's time and about 3× its typical error. "Exact" in older handoffs means exact against our own hardware-rounding oracle, not against Model or the simulator.

## 1. Summary

| Metric | Legacy headset (V69, QNN) | Frame main tree (V75) | Frame run-list candidate (unmerged) |
|---|---:|---:|---:|
| Warm inference | **9.71 ms** | **26.56 ms** | **22.79 ms** |
| Ratio to Model | 1.0× | 2.74× | 2.35× |
| Median f32 mean error vs simulator | 0.0402 | 0.1225 | same oracle as main |
| Median f32 \|bias\| vs simulator | 0.0016 | 0.0840 | same |
| Median f32 p95 error vs simulator | 0.1450 | 0.5062 | same |
| Cases at or below Model's mean error | n/a | 0 of 27 | same |
| Output hash matches | simulator on 5 of 27 cases (u8) | hw oracle on smoke and random | hw oracle on smoke and random |

The candidate runs the same semantics as main: its outputs hash to the same hardware oracle on smoke and random.

## 2. Measurement provenance

| Item | Source | Notes |
|---|---|---|
| Model | `models/primary/context.bin`, QNN 2.19, SHA256 `2686e657…`; 5,068 scheduled commands | Primary only. Alternate is not analyzed here. |
| Frame main-tree timing | `frame_reports/four_primary.json` (2026-11-02), `QDS_PREPARED_THREADS=4`, variant `l2fast4` | **Not re-run today.** No engine source changed since this report, based on file modification times. Includes `-O3 -ffast-math`. |
| Frame candidate timing | `frame_reports/claim_primary.json` (2026-11-02), `/tmp/frame-i-merge`, `QDS_QNN_CLAIM=1` | Primary only; alternate has no captured run list. Fusion off. |
| Model timing and outputs | `.work/model-qnn-meta-20261030/full-validation*` (2026-10-30) | Vendor `libhexagon` RPC time, excluding model creation and file I/O. 3 calls per case, deterministic. |
| Reference ("simulator") | `cpu-f32.raw`, `cpu-u8.raw` in each Model case directory | QAIRT 2.50 x86 simulator semantics. The host control build reproduced all 27 of 27 byte for byte. |
| Frame error, smoke and random | Hash equals `recovered/primary/hw_*.json`, the oracle Frame passed | Directly tied to Frame runs. |
| Frame error, other 25 cases | Host scalar build with `-DNPRG2_HW_ROUNDING=1` | **Host only**, not run on Frame. Inputs regenerated with `tools/stereo_probes.py`; all 27 input hashes match what Model ran. |

All inputs are synthetic: textured shifts, flat images, ramps and noise. **No real stereo frames have been compared.** Output units are the model's float output, which ranges up to about 32 on these cases.

## 3. Timing detail

| Build | Warm median | Min–max | First run | Runs |
|---|---:|---:|---:|---:|
| Main, 3 workers | 27.50 ms | 27.30–27.61 | 133.8 | 5 |
| Main, 4 workers | 26.56 ms | 26.49–26.75 | 144.3 | 5 |
| Candidate, run lists + 4 workers + claim parts | 22.79 ms | 22.64–22.85 | 127.2 | 5 |
| Model | 9.71 ms | 9.15–10.16 | n/a | 54 calls after each case's first |

**Main tree, 4 workers, last random run**

| Phase | ms |
|---|---:|
| Handler run | 24.05 |
| Executor dispatch | 0.785 |
| Reused preflight | 0.097 |
| Convolution handlers (opcode 6) | 12.55 |
| Weights to VTCM (opcode 4) | 1.64 |
| All other handlers | 9.86 |

Near-conv phases, summed over thread parts (CPU time, not wall): staging 1.40, MAC 5.07, setup 1.53, output 1.06 ms.

Parallel jobs: conv 662, elementwise 374, data movement 304.

**Attribution update:** the blanket summed-CPU label above must not be carried into later HMX measurements. Section 11.3 audits the actual timing paths and distinguishes caller elapsed time from summed fallback worker time.

**Candidate lanes** (from `HANDOFF_SESSION_20261102H.md`): main wall 21.1 ms, main busy 14.05 + wait 6.6; worker busy 11.9 + wait 8.6. Wait attribution charges main's wait to the 80-plane dynamic-weight preparation chain. The schedule model in `candidate/runlist_model/` gives, from a 20.4 ms modeled baseline: all conv ×0.5 → 15.3; conv and worker ×0.5 → 10.9. These are model projections, not measurements.

**Handlers, top 30 by main-tree run time** (share of 24.05 ms; remaining 46 handlers total 2.76 ms)

| Op/var | Variant | Calls | Main ms | Share | µs/call | Candidate ms (lane) |
|---|---|---:|---:|---:|---:|---|
| 6/33 | `ConvLayer_s1.opt@CB*2.Xwb.Fi.t` | 629 | 7.775 | 32.3% | 12.4 | 7.448 (main) |
| 6/32 | `ConvLayer_s1.opt@CB*2.Xpb.Fi.t` | 217 | 1.721 | 7.2% | 7.9 | 1.600 (main) |
| 6/35 | `ConvLayer_s1.opt@CH*2.Xwb.Fi.t` | 70 | 1.477 | 6.1% | 21.1 | 1.630 (main) |
| 4/30 | `ConvLayer.opt.weights_to_vtcm@Xwb.xwb.` | 64 | 1.349 | 5.6% | 21.1 | 0.190 (main) |
| 6/36 | `ConvLayer_s2.opt@CB*2.Xwb.Fi` | 80 | 1.284 | 5.3% | 16.1 | 1.396 (main) |
| 7/31 | `ConvLayer_SpaceToDepth_stride2.tcm@CB*2.s4` | 174 | 0.752 | 3.1% | 4.3 | 0.929 (worker) |
| 24/51 | `ResizeNear_slice.tcm_special@CB*2.fi*2` | 107 | 0.652 | 2.7% | 6.1 | 0.736 (worker) |
| 18/55 | `Transpose.2D@CB*2.t` | 82 | 0.501 | 2.1% | 6.1 | 0.543 (worker) |
| 5/24 | `ConvLayer.opt.inconv_im2col.tcm@CB.FB.s4.t` | 80 | 0.459 | 1.9% | 5.7 | 0.504 (worker) |
| 10/13 | `Add.tcm@CB*3` | 170 | 0.395 | 1.6% | 2.3 | 1.039 (worker) |
| 16/45 | `ForceFormat_Flat_c2f@FH.CH` | 10 | 0.363 | 1.5% | 36.3 | 0.372 (worker) |
| 33/66 | `invscale_to_qi32@fs.ff` | 80 | 0.342 | 1.4% | 4.3 | 0.370 (worker) |
| 35/23 | `ConvLayer.opt.convert_weights_to_signed.packed@xpb.FB.t.fs.t` | 80 | 0.340 | 1.4% | 4.2 | 0.364 (worker) |
| 18/56 | `Transpose.2D@CH*2.t` | 18 | 0.325 | 1.4% | 18.1 | 0.346 (worker) |
| 26/8 | `@Fill` | 62 | 0.297 | 1.2% | 4.8 | 0.312 (main) |
| 6/34 | `ConvLayer_s1.opt@CH*2.Xpb.Fi.t` | 40 | 0.296 | 1.2% | 7.4 | 0.402 (main) |
| 16/43 | `ForceFormat_Crouton_f2c@CH.FH` | 11 | 0.280 | 1.2% | 25.5 | 0.304 (worker) |
| 8/70 | `linearclip_simple_rq@CB*2.fi` | 160 | 0.235 | 1.0% | 1.5 | 1.041 (worker) |
| 16/49 | `Reshape.hw@CB*2` | 80 | 0.235 | 1.0% | 2.9 | 0.289 (worker) |
| 16/41 | `ForceFormat_Crouton@CB*2` | 70 | 0.233 | 1.0% | 3.3 | 0.265 (worker) |
| 14/74 | `mul_op_crt@CH*3.fi` | 50 | 0.220 | 0.9% | 4.4 | 0.239 (worker) |
| 39/68 | `linearclip@CH.CB.fi` | 40 | 0.217 | 0.9% | 5.4 | 0.257 (worker) |
| 32/21 | `ConvLayer.opt.conv_scale_from_weights@ff.FB` | 80 | 0.203 | 0.8% | 2.5 | 0.234 (worker) |
| 9/71 | `linearclip_simple_rq@CH*2.fi` | 54 | 0.202 | 0.8% | 3.7 | 0.218 (worker) |
| 13/59 | `addsub_op@CH*2.fH.fi` | 60 | 0.197 | 0.8% | 3.3 | 0.221 (worker) |
| 16/65 | `flat_to_vtcm@Ff.ff.` | 160 | 0.194 | 0.8% | 1.2 | 0.243 (worker) |
| 4/29 | `ConvLayer.opt.weights_to_vtcm@Xpb.xpb.` | 92 | 0.188 | 0.8% | 2.0 | 0.247 (main) |
| 16/42 | `ForceFormat_Crouton_f2c@CB.FB` | 10 | 0.188 | 0.8% | 18.8 | 0.205 (worker) |
| 34/20 | `ConvLayer.opt.combine_scales@ff*3.t` | 80 | 0.185 | 0.8% | 2.3 | 0.209 (worker) |
| 2/2 | `*InputSlicePad@FB.s4*3.t.s4*3.` | 80 | 0.184 | 0.8% | 2.3 | 0.210 (worker) |

**Workload.** 926 u8 convs (23.34 G MACs) and 110 u16 convs (0.30 G MACs), counting logical MACs from the graph. Handoff H measured 6/33 at 21.7 G padded MACs in 7.37 ms, with the MAC phase about 5 ms. Program hot tables are about 235 KB; run-time heap 4.95 MB. VTCM is 8 MB.

**Model op profile** (`profile-validation/primary/smoke/profile.json`, profiled run, top 12 of 201 logical ops). These are logical QNN op aggregates, not compiled tiles or a timeline, and overlap sums aren't additive wall time. Profiling slows the call to about 175–190 ms. Total node cycles 4,650,158, wait 23,979. Use these for relative comparison only.

| Op | Cycles | Wait | Overlap | Resource mask |
|---|---:|---:|---:|---:|
| boltnn_quantized_conv2d_nhwc_tensor:OpId_196 | 342,567 | 986 | 193,564 | 7 |
| quantized_decomposed_dequantize_per_tensor_default_333:OpId_841 | 280,900 | 11 | 258,546 | 5 |
| boltnn_quantized_bmm_tensor:OpId_347 | 230,465 | 6,997 | 180,363 | 7 |
| boltnn_gather_tensor:OpId_362 | 213,646 | 0 | 34,401 | 1 |
| Output OpId_3 | 199,416 | 40 | 17,590 | 5 |
| boltnn_permute_to_boltnn_tensor_1:OpId_199 | 189,817 | 0 | 54,785 | 1 |
| boltnn_convert_tensor:OpId_741 | 171,886 | 43 | 105,824 | 1 |
| boltnn_quantized_conv2d_nhwc_tensor_10:OpId_329 | 118,556 | 724 | 44,314 | 6 |
| boltnn_quantized_conv2d_nhwc_tensor_69:OpId_754 | 115,541 | 302 | 115,541 | 6 |
| boltnn_quantized_conv2d_nhwc_tensor_3:OpId_248 | 113,516 | 373 | 90,482 | 6 |
| boltnn_quantized_conv2d_nhwc_tensor_74:OpId_760 | 110,809 | 247 | 59,385 | 7 |
| boltnn_quantized_conv2d_nhwc_tensor_7:OpId_265 | 107,180 | 1,373 | 78,079 | 7 |

## 4. Error against the simulator: float output

Every number is measured against the simulator reference. "FIX" is the bias-correction experiment in section 6.

Group medians:

| Group | Model MAE / \|bias\| / p95 | Frame hw | FIX |
|---|---|---|---|
| Textured and random (22 cases) | 0.0409 / 0.0040 / 0.1475 | 0.1158 / 0.0718 / 0.4637 | 0.0783 / 0.0100 / 0.2900 |
| Flat and smoke (5 cases) | 0.00008 / 0.00008 / 0.0000 | 1.0820 / 1.0810 / 2.6759 | 0.6142 / 0.4925 / 3.5108 |

Cases at or below Model:

| Build | MAE | \|bias\| | p95 | max |
|---|---:|---:|---:|---:|
| Frame hw | 0 | 0 | 0 | 5 |
| FIX | 0 | 5 | 0 | 5 |

Per case (each group: MAE, signed bias, p95, max):

| Case | Q MAE | Q bias | Q p95 | Q max | HW MAE | HW bias | HW p95 | HW max | FIX MAE | FIX bias | FIX p95 | FIX max | HW/Q | FIX/Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 0.00007 | +0.00007 | 0.0000 | 0.004 | 1.47802 | +1.44814 | 4.6044 | 6.229 | 0.45647 | +0.13410 | 1.3173 | 2.740 | 19807.2× | 6117.3× |
| same | 0.02130 | +0.00055 | 0.0612 | 0.452 | 0.06061 | +0.05451 | 0.1575 | 0.789 | 0.03060 | +0.00629 | 0.0837 | 0.699 | 2.8× | 1.4× |
| w+2 | 0.00855 | -0.00015 | 0.0250 | 0.329 | 0.02117 | +0.01778 | 0.0537 | 0.221 | 0.01478 | +0.00101 | 0.0387 | 0.201 | 2.5× | 1.7× |
| w-2 | 0.28492 | +0.02185 | 0.9686 | 22.312 | 0.77071 | -0.53519 | 2.1610 | 25.995 | 0.70347 | -0.43716 | 2.0735 | 25.845 | 2.7× | 2.5× |
| h+2 | 0.04738 | +0.01436 | 0.1650 | 3.426 | 0.11807 | -0.06329 | 0.4787 | 2.831 | 0.09625 | +0.04836 | 0.3750 | 3.603 | 2.5× | 2.0× |
| h-2 | 0.04755 | +0.00155 | 0.1612 | 1.936 | 0.16074 | -0.10441 | 0.6887 | 3.492 | 0.11106 | +0.05417 | 0.3812 | 3.516 | 3.4× | 2.3× |
| w+4 | 0.00687 | +0.00006 | 0.0225 | 0.246 | 0.02453 | +0.02211 | 0.0587 | 0.237 | 0.01267 | -0.00041 | 0.0350 | 0.199 | 3.6× | 1.8× |
| w-4 | 0.04709 | -0.00102 | 0.1600 | 1.651 | 0.12255 | -0.08296 | 0.4549 | 2.327 | 0.10360 | +0.04801 | 0.3637 | 1.549 | 2.6× | 2.2× |
| h+4 | 0.03875 | +0.00633 | 0.1425 | 1.430 | 0.10515 | -0.06981 | 0.4174 | 2.673 | 0.06984 | -0.00676 | 0.2600 | 3.408 | 2.7× | 1.8× |
| h-4 | 0.03966 | -0.00032 | 0.1475 | 1.037 | 0.14233 | -0.11637 | 0.6274 | 2.002 | 0.08417 | -0.00873 | 0.3337 | 1.437 | 3.6× | 2.1× |
| w+8 | 0.01036 | +0.00042 | 0.0337 | 0.392 | 0.05050 | +0.04854 | 0.1050 | 0.412 | 0.01898 | +0.00016 | 0.0512 | 0.757 | 4.9× | 1.8× |
| w-8 | 0.04646 | -0.00596 | 0.1600 | 2.712 | 0.16645 | -0.10277 | 0.7424 | 3.020 | 0.08546 | +0.00536 | 0.3112 | 4.037 | 3.6× | 1.8× |
| h+8 | 0.04623 | -0.01406 | 0.1712 | 1.211 | 0.13485 | -0.08697 | 0.5187 | 1.341 | 0.07783 | +0.01357 | 0.2900 | 1.559 | 2.9× | 1.7× |
| h-8 | 0.04903 | +0.00557 | 0.1850 | 0.850 | 0.14474 | -0.11627 | 0.5949 | 2.180 | 0.09554 | +0.01189 | 0.3899 | 1.630 | 3.0× | 1.9× |
| w+16 | 0.02330 | -0.00092 | 0.0662 | 1.251 | 0.06146 | +0.05689 | 0.1375 | 1.281 | 0.03345 | -0.00863 | 0.0887 | 1.469 | 2.6× | 1.4× |
| w-16 | 0.04103 | +0.00805 | 0.1450 | 0.941 | 0.10525 | -0.05607 | 0.4249 | 1.384 | 0.07828 | +0.03492 | 0.2900 | 1.115 | 2.6× | 1.9× |
| h+16 | 0.06292 | +0.02446 | 0.2425 | 1.214 | 0.14050 | -0.09642 | 0.5462 | 2.398 | 0.07792 | +0.00387 | 0.2825 | 3.090 | 2.2× | 1.2× |
| h-16 | 0.04024 | -0.00682 | 0.1475 | 0.895 | 0.11793 | -0.06487 | 0.4724 | 1.447 | 0.07835 | +0.02188 | 0.2850 | 2.493 | 2.9× | 1.9× |
| w+32 | 0.04125 | -0.00018 | 0.1137 | 1.607 | 0.11365 | +0.10905 | 0.2312 | 1.796 | 0.05221 | -0.00117 | 0.1350 | 2.123 | 2.8× | 1.3× |
| w-32 | 0.04080 | +0.00603 | 0.1475 | 1.851 | 0.11284 | -0.05445 | 0.5162 | 2.173 | 0.09460 | +0.04034 | 0.3512 | 1.511 | 2.8× | 2.3× |
| h+32 | 0.04171 | -0.00193 | 0.1550 | 2.651 | 0.12827 | -0.07376 | 0.5062 | 2.536 | 0.08823 | -0.00130 | 0.3537 | 2.441 | 3.1× | 2.1× |
| h-32 | 0.04086 | +0.00252 | 0.1525 | 0.867 | 0.10541 | -0.05981 | 0.4324 | 1.669 | 0.08638 | +0.05131 | 0.3350 | 1.374 | 2.6× | 2.1× |
| flat_h0_id | 0.00008 | +0.00008 | 0.0000 | 0.004 | 1.08201 | +1.08098 | 2.1647 | 3.085 | 0.61417 | +0.49254 | 3.5108 | 5.362 | 13561.6× | 7697.9× |
| flat_h0_zero | 0.00008 | +0.00008 | 0.0000 | 0.004 | 1.08201 | +1.08098 | 2.1647 | 3.085 | 0.61417 | +0.49254 | 3.5108 | 5.362 | 13561.6× | 7697.9× |
| flat_h16_id | 0.00019 | +0.00019 | 0.0025 | 0.004 | 1.06460 | +1.06333 | 2.6759 | 4.568 | 0.61685 | +0.49279 | 3.5508 | 5.870 | 5678.9× | 3290.5× |
| flat_h16_zero | 0.00019 | +0.00019 | 0.0025 | 0.004 | 1.05527 | +1.05273 | 3.3883 | 5.519 | 0.62049 | +0.49293 | 3.5945 | 6.827 | 5542.9× | 3259.2× |
| random | 0.03307 | +0.00540 | 0.0950 | 5.709 | 0.09240 | +0.08400 | 0.2062 | 5.396 | 0.06206 | +0.01121 | 0.1500 | 6.338 | 2.8× | 1.9× |

**On low-texture inputs Model has essentially no error, while Frame hw has a bias above +1.**

## 5. Error against the simulator: u8 output

Each cell: MAE / max / percent of values differing. Model is byte-exact on 5 of 27 cases.

| Case | Model | Frame hw | FIX |
|---|---|---|---|
| smoke | 0.000 / 0 / 0.0% | 2.086 / 16 / 83.7% | 1.538 / 20 / 78.8% |
| same | 0.702 / 9 / 56.6% | 1.108 / 11 / 71.8% | 0.958 / 8 / 67.0% |
| w+2 | 0.452 / 5 / 41.7% | 0.953 / 7 / 67.7% | 0.827 / 7 / 62.2% |
| w-2 | 1.234 / 72 / 70.8% | 2.169 / 93 / 80.8% | 1.875 / 80 / 77.8% |
| h+2 | 0.825 / 8 / 63.1% | 1.351 / 12 / 76.2% | 1.031 / 11 / 69.9% |
| h-2 | 0.831 / 6 / 63.6% | 1.465 / 11 / 78.1% | 1.055 / 10 / 70.9% |
| w+4 | 0.270 / 3 / 26.1% | 0.762 / 6 / 61.1% | 0.548 / 5 / 48.3% |
| w-4 | 0.837 / 8 / 63.5% | 1.465 / 11 / 78.4% | 1.099 / 11 / 71.5% |
| h+4 | 0.789 / 6 / 61.7% | 1.344 / 12 / 76.1% | 0.987 / 11 / 68.7% |
| h-4 | 0.785 / 8 / 61.6% | 1.403 / 11 / 76.7% | 1.017 / 8 / 69.3% |
| w+8 | 0.407 / 5 / 37.9% | 1.127 / 6 / 72.5% | 0.746 / 6 / 59.8% |
| w-8 | 0.822 / 6 / 63.0% | 1.510 / 12 / 78.3% | 1.045 / 11 / 70.2% |
| h+8 | 0.814 / 6 / 62.6% | 1.371 / 11 / 76.8% | 0.997 / 9 / 68.9% |
| h-8 | 0.823 / 6 / 62.8% | 1.462 / 11 / 77.4% | 1.023 / 9 / 69.4% |
| w+16 | 0.628 / 8 / 53.7% | 1.211 / 8 / 74.7% | 0.891 / 10 / 65.8% |
| w-16 | 0.804 / 8 / 62.4% | 1.363 / 10 / 76.7% | 1.020 / 11 / 69.6% |
| h+16 | 0.850 / 7 / 63.9% | 1.422 / 16 / 77.5% | 1.012 / 17 / 69.3% |
| h-16 | 0.794 / 6 / 61.9% | 1.372 / 10 / 76.7% | 1.014 / 7 / 69.3% |
| w+32 | 0.829 / 9 / 63.4% | 1.212 / 10 / 74.3% | 1.009 / 11 / 69.5% |
| w-32 | 0.798 / 6 / 62.1% | 1.368 / 10 / 76.1% | 1.035 / 13 / 70.0% |
| h+32 | 0.792 / 6 / 61.9% | 1.364 / 9 / 76.8% | 1.001 / 10 / 69.2% |
| h-32 | 0.793 / 5 / 61.9% | 1.347 / 9 / 76.5% | 1.005 / 7 / 69.3% |
| flat_h0_id | 0.000 / 0 / 0.0% | 2.093 / 11 / 84.8% | 1.425 / 13 / 76.2% |
| flat_h0_zero | 0.000 / 0 / 0.0% | 2.093 / 11 / 84.8% | 1.425 / 13 / 76.2% |
| flat_h16_id | 0.000 / 0 / 0.0% | 2.093 / 11 / 84.8% | 1.425 / 13 / 76.2% |
| flat_h16_zero | 0.000 / 0 / 0.0% | 2.093 / 11 / 84.8% | 1.425 / 13 / 76.2% |
| random | 0.727 / 13 / 58.9% | 1.110 / 20 / 72.1% | 1.013 / 23 / 69.3% |

Error of other experiments (mean vs the hw oracle on Frame, from `frame/build/precision-g/`; the reference differs from the tables above): cost-volume step 2, 25.3–25.7 ms in older three-worker runs, smoke f32 mean 0.052, random 0.921; step 8, 26.6–26.9 ms, smoke 0.527, random 1.086. Both are far outside Model's error and show a poor trade.

## 6. Root cause so far

**Frame HMX rounds differently.** Per `HMX_NATIVE_ROUNDING.md`, V75 `cvt.ub` computes `clip(((2q+1)*M + 2^14) >> 15)` while the simulator computes `clip((q*M + 2^13) >> 14)`, so conv output is 0 or +1 higher, and exactly equal only when `M=1024`. A 300-case rerun of `bench/hmx_cvt_random_sim.c` on the instruction-set simulator gave the same 0 or +1 difference from the simulator formula on **V73 and V75**, with identical results on both.

**Why Model differs is unknown.** Model doesn't show this bias: on these cases its signed error is near zero. `HMX_NATIVE_ROUNDING.md` infers that Model differs from the simulator because of this rule, and these results contradict that inference. Possibilities are that V69 silicon rounds differently, or that something else is going on. The test bench can't settle it: this HMX instruction form requires `-mv73` or higher in the assembler, and a V73 ELF won't load under `hexagon-sim -mv69`. Model's own differences from the simulator (58.9% of u8 values on random) also have no identified source. The V69 library's own HMX leaves are named `hmx_v73_*`.

**Bias-correction experiment (FIX).** Subtract half a `q` unit from the accumulator before conversion: u8 `a -= 1<<(pre-1)`; u16 only when representable in whole bias units. Scalar host model only, `scripts/host_dump_and_bias_fix.patch`.
- Median \|bias\| fell from 0.0840 to 0.0136, and median MAE from 0.1225 to 0.0855.
- **It doesn't meet the goal:** 0 of 27 cases reach Model's MAE. Flat cases stay around 0.6 with a bias of +0.49.
- Not implemented in the HMX or HVX paths. Dynamic-weight convs compute bias at run time, so they would need it there.

## 7. Options for the review

Historical starting options. Results from executing these investigations are in section 11; several are now measured or rejected.

Not ranked as conclusions; all need measurement.

1. **Simulator-rounding convs everywhere.** Zero error against the simulator, so it passes by construction. Older comparison: `l2hmx` 117.9 ms vs hw 87.6 ms at that stage, with u8 convs at half the native rate. Measure this on the current engine first, since it's the guaranteed-correct reference point and may be much closer now.
2. **Selective exactness.** Use simulator rounding only on the convs that drive error, and native rate elsewhere. Needs per-layer error sensitivity, which the host build can produce cheaply.
3. **Better compensation.** Per-channel offsets beyond FIX, or adjusting downstream quantization.
4. **Exploit `M=1024`.** Native conversion is exact there, with M applied in HVX. This is the v6 approach, so it's largely option 1's cost, unless a cheaper combined form exists.
5. **Find what Model actually does.** Disassemble the conv leaves in `libQnnHtpV69.so` and the requantize they use, then compare with the V75 path. Could change every option above.
6. **Continue pure speed work once a passing mode exists:** merge the candidate, then 6/33 HMX at about 7.4 ms, the weight-preparation chain, and broad 2× worker improvement.

Separately, test real captured stereo pairs. The Frame hw bias is largest on flat, low-texture input (above +1), and the room-reconstruction notes report a positive disparity bias and doming on low-texture walls. Whether they are connected is **an untested hypothesis**, stereo yaw drift is already a known separate cause, and it hasn't been confirmed which rounding mode produced those captures.

## 8. Proposed gate

The user's criterion needs one concrete definition. Proposed and **not yet confirmed**: for each of the 27 cases plus real pairs, f32 MAE, \|signed bias\| and p95 at or below Model's, with max reported but not gated, because Model's own max reaches 22.3 on `w-2`. A small margin may be needed for near-zero cases, where Model's p95 is 0. Passing `ab.sh` alone only proves a build matches its own oracle.

## 9. Known issues

- `EXECUTOR_REVIEW_20261102.md` lists three executor correctness issues still open.
- The candidate tree and its patches lived only in `/tmp`; copies of the patches are in `candidate/`, but the `QDS_QNN_CLAIM` state on `/tmp/frame-i-merge` is uncommitted. Details in `HANDOFF_SESSION_20261102H.md`.
- Short interval `0..1` shows a known 16-byte pool-1 difference under `QDS_POOL1_NOCOPY`. Don't report it as passing.
- Fast math is opt-in and exact only on the tested cases.
- The disk was nearly full during this work (down to 338 MB free, recovering on its own); `/private/tmp` holds about 51 GB. Two shell runs died without output around the dip; low disk is the likely cause but wasn't proven. Check free space before long runs.

## 10. Reproducing

From `depth-model/`. `X` is any work directory holding `src/` (a copy of `frame/*.c,*.h` and `frame/native_program/*.c,*.h,*.inc` with the patch applied, plus the headers in `frame/layout_audit/` and `frame/native_*/`), `bin/` and `inputs/`:

```sh
python3 tools/stereo_probes.py make "$X/inputs"
sh frame/perf_baseline_20261105/scripts/build.sh "$X"
python3 frame/perf_baseline_20261105/scripts/sweep.py "$X" emu hw fix   # about 2.7 s per case per build
python3 frame/perf_baseline_20261105/scripts/eval.py "$X"               # writes metrics.json
sh frame/bench/ab.sh -r 3 -t name l2fast4                                # Frame timing; one device client at a time
```

Everything cited is in `frame/perf_baseline_20261105/`: `error_metrics_primary.json` (all per-case metrics, including Frame against Model), `frame_reports/`, `scripts/`, `candidate/`.

## 11. Follow-up findings — 2026-11-05 (current status)

**Best validated accurate primary candidate: 42.014 ms warm median. The requested 4× improvement means about 10.5 ms and is NOT achieved.** Confidence is high in the measured bottlenecks and the tested accuracy result, but there is no demonstrated implementation that reaches 10.5 ms.

The user clarified that exact u8 behavior is not a requirement: any implementation is acceptable if it achieves a substantial speedup while keeping error below Model. “Accurate” below means the working gate of f32 MAE, absolute signed bias, and p95 versus the **QAIRT 2.50 simulator** at or below Model on each of the 27 synthetic cases. Max and the separate u8 output are reported separately. This is numerical agreement with a reference, not a real-world depth-accuracy claim. No real stereo-pair validation was added. The reference was not silently changed after obtaining QAIRT 2.25.

Sections 1–10 preserve the starting baseline. In particular, their hardware-rounding speed results remain valid historical results; they are not regressions caused by the accurate experiments.

### 11.1 Accuracy isolation and measured performance

| Experiment | Warm inference | Accuracy / disposition |
|---|---:|---|
| Historical native-rounding main / run lists | 26.56 / 22.79 ms | Higher error than Model; not an accuracy-passing comparison |
| Current all-simulator-rounding, four workers | ~61.5 ms | Smoke/random match simulator; u16 HMX disabled in this mode |
| Simulator u8 + native u16 | ~46 ms initially | Large bias removed; all 27 host cases pass f32 MAE, absolute bias, p95 |
| Same mode, original versus streaming staging, paired | **45.389 → 43.976 ms** | 3.1% measured improvement; outputs unchanged |
| Streaming, fusion disabled | ~43.7–43.8 ms | Smoke/random hashes unchanged; only a small difference, not a large demonstrated gain alone |
| Streaming + QNN run lists + claim-based helpers | **42.014 ms** | Best validated candidate; 27 cases × 2 runs pass mixed-host hashes on Frame |

The mixed host sweep's median f32 MAE and absolute bias are both **0.0000050573**, with median p95 **0**. All 27 cases beat or equal Model on those three metrics against the original 2.50 reference; max passes only **22/27**. The initial device test covered smoke/random; the later streaming and combined run-list tests extended byte-identical agreement with this host oracle to **all 27 cases on Frame**, both output heads.

Evidence, relative to `frame/`:

- `build/ab/investigation-exact-20261105/`: exact build, flags, short check and primary report.
- `build/ab/investigation-mixed-20261105/`: mixed build, `metrics_mixed.json`, `host_hashes.json`, `mixed_host_checked.json` (6/6 smoke/random runs).
- `build/ab/investigation-stream-real-20261105/`: `control_paired.json`, `stream_paired.json`, `all27_host_checked.json` (54 runs), 27-case host hashes and checking wrapper.
- `build/ab/investigation-runlist-mixed-20261105/`: `confirm.json` (14 smoke/random runs, warm median 42.0140877 ms), `all27_host_checked.json` (54 runs, warm median 42.0352018 ms), `summary.json`, engine/bootstrap, `build-env.json`, `source.tar.gz`, and `accuracy-streaming.patch` against the earlier run-list source.

**What changed in code:** `NPRG2_HW_ROUNDING_U16` independently controls u16 rounding and defaults to `NPRG2_HW_ROUNDING`, preserving existing defaults. The passing mode sets only `NPRG2_HW_ROUNDING_U16=1`. It retains u16 HMX while using simulator rounding for u8. This isolates native u8 rounding as the cause of the large measured error in these tests; it does **not** establish Model's own per-layer rounding rule.

The original exact-mode **52.5 ms u16 MAC figure was summed worker time**, not inference wall time. Its actual inference was ~61.5 ms. It must not be compared directly with the old ~24 ms native-rounding inference. Switching to mixed mode recovered about 15 ms wall time by retaining native u16 HMX.

The successful staging change replaces a dynamically indexed `HVX_Vector V[10]` temporary with streamed adjacent vector loads. Disassembly of the old helper showed a **1,536-byte stack frame and vector stack stores/reloads**; the replacement was inlined into a helper with an **80-byte frame**. This is an observed compiler/code-generation effect, and the paired timing above measures its benefit.

### 11.2 Current source/build state and reproduction

The combined 42 ms run-list candidate remains **isolated, not merged wholesale into the main checkout**. Its working source is `/tmp/frame-speed-runlist/depth-model`; the exact source snapshot is preserved in `build/ab/investigation-runlist-mixed-20261105/source.tar.gz`. The main working tree contains the independent-u16 flag and successful streaming helper change, alongside substantial pre-existing edits. No commit, reset, production deployment, or blanket cleanup was performed.

The authoritative build configuration is `build-env.json` beside that source archive. Important settings:

- Four workers, HMX, reference executor, L2 cache mode 1, profiling enabled.
- `QDS_PREPARED_FUSE=0`, `QDS_PREPARED_DMA_ASYNC=1`.
- Extra flags: `-DNPRG2_HW_ROUNDING_U16=1 -DNPRG2_SLIDE3=1 -DNPRG2_QNN_RUNLISTS=1 -DQDS_QNN_CLAIM=1 -DQDS_PREPARED_DMA_MODE=0 -O3`.
- These accurate candidate measurements did **not** enable `-ffast-math` or `QDS_POOL1_NOCOPY`.

`ab.sh` still expects the emulator oracle unless hardware rounding is selected. It therefore labels mixed results FAILED even when they correctly match the mixed host oracle. Do not reinterpret such a failure without checking the actual hashes and execution/cleanup guards. The experimental `device_check.py` wrapper substitutes the explicit mixed host hashes for acceptance; it does not assert simulator-exactness.

Example current-device validation, from the remote `frame-dsp/depth-model/wave1-20261101` directory (one Frame client at a time):

```sh
/home/steamos/frame-dsp/depth-model/.venv/bin/python \
  frame/build/ab/investigation-stream-real-20261105/device_check.py \
  --model primary --all-cases --repeat 2 \
  --engine frame/build/ab/investigation-runlist-mixed-20261105/eng-runlist.elf \
  --bootstrap frame/build/ab/investigation-runlist-mixed-20261105/boot-runlist.elf \
  --prepared-program frame/native_program/examples/primary-v2.nprg \
  --report results/ab/investigation-runlist-mixed-20261105/all27_host_checked.json
```

Local scalar reproduction uses `/tmp/frame-rounding-ablation` and the baseline scripts. `scripts/build.sh` now propagates compiler failures, saves build logs, resolves its output directory, and includes a `mixed` target. `scripts/eval_variant.py WORK mixed` evaluates both heads against simulator and Model. Host sensitivity runners took ~22 s per full-model invocation in the later session, rather than the earlier ~2.7 s estimate; do not budget sweeps from the old estimate alone.

The archives/reports in `frame/build/` and SDK in `.work/` are local artifacts and may be ignored by git. Preserve them explicitly before deleting temporary/build directories; a path in this document does not imply a committed artifact.

### 11.3 Corrected timing attribution: HMX is active

An explicit compile-gated path-count probe counted **926 u8 HMX path-3 commands plus 110 native16/slide path-5 commands**, accounting for **all 1,036 convolutions**. There was no near/scalar fallback in this measured primary run. Legacy `direct_hmx=false` and zero legacy HMX counters do not describe prepared-executor callbacks and cannot be used to infer that HMX is inactive.

The earlier blanket label “summed worker CPU time” is **not correct for every phase field**:

- `hmx_conv()` times staging and quant/output around the caller's `nprg2_parallel_for` joins; HMX callbacks do not add their individual times. These are caller-elapsed phase times, summed across sequential convolution commands.
- The near/HVX fallback accumulates per-part timings and can report summed worker time. The all-exact u16 52.5 ms number belongs to that category.
- The counters establish that the mixed HMX build's phase values belong to the first category.

Representative final warm combined run: total DSP **41.996 ms**, main-list wall **38.428 ms**, main convolution handlers **30.810 ms**, worker handlers **11.775 ms**. Worker/main sums overlap and are not additive inference time. HMX phases: staging **13.277**, MAC **4.386**, setup **5.349**, output **4.664 ms**. DSP read/write phases were **1.731 / 0.201 ms**. Host upload/download (~0.074 / 0.433 ms in that run) are separate from `dsp_ms` and should not be added twice.

The diagnostic smoke capture also recorded staging/MAC/setup/output ~13.57/4.39/6.52/4.88 ms. These are different captures/conditions, not a replacement for the final warm values. Evidence and instrumentation definitions: `build/ab/investigation-runlist-mixed-20261105/path_count_probe/`.

**Conclusion supported by measurement:** arithmetic issue/execution is a small fraction of the accurate engine's time. Preparation, layout, setup and output handling dominate. This does not prove that all of those costs can be eliminated, nor that firmware interference is absent.

### 11.4 Rejected experiments and recovery

| Experiment | Result | Disposition / evidence |
|---|---|---|
| Aligned quad loads replacing partial-load assembly | Candidate smoke/random medians 48.27/47.85 ms vs control 46.49/46.45; all 14 runs match mixed hashes | Reverted; `build/ab/investigation-quad-20261105/`, including saved patch |
| Two workers for exact HMX staging/output, four-worker engine otherwise | ~57.0–57.8 ms vs ~46 ms; all 6 outputs match mixed hashes | Reverted; `build/ab/investigation-parts-20261105/` |
| First fused 3×3 tap-staging prototype | Full27 exact but ~44.17 ms vs ~42.13 ms control | Rejected; saved in `path_count_probe/` |
| Revised fused tap-staging prototype | Paired warm **42.953 vs 42.156 ms**; full27 **42.899 vs 42.134 ms** | Rejected and unmerged; full27 exact, clean exits, VTCM restored |

The fused prototype tries to reuse source vectors across adjacent taps while retaining the exact A-tile representation. Per full 32-pixel tile/channel block, unique 3×3 support is **3×34×32 = 3,264 bytes**, versus about **9×32×32 = 9,216 bytes** of repeated source reads; the **18 KiB A-tile writes remain**. Thus a ~2.8× read reduction is not a 2.8× staging-time promise. The revised code fixed a y/x eligibility-index bug, distributed `(tile, channel-block)` work across helpers, and moved the inverse K map out of repeated per-part setup. Its stack frame fell from 640 to 384 bytes, but staging/setup overhead still outweighed saved loads. Both versions ultimately passed all 27 cases; neither is adopted.

A short quad-load check passed its output comparison but stalled during DSP session shutdown. Guarded recovery verified worker exit/VTCM restoration but could not return the service to idle. The user authorized a Frame reboot; after reboot the module was reloaded with `size_mb=128`, and `chunk_map.py` reported `n_bad=0`. Subsequent validation and shutdown succeeded. This was not a measured speed gain or a proven kernel-caused failure. Later fused experiments required no reboot; their remote diagnostic files were removed, worker exit errors were zero, and VTCM fingerprints matched. Credentials are intentionally not recorded here.

A standalone wide-output ISA-simulator probe and instruction-assembly checks did not produce a direct accurate replacement for native `cvt.ub`. The tested wide-output behavior combines lanes; attempted plain `cvt.uh` and `.after.uw` forms were rejected by the assembler. This rules out those attempted forms, not every possible ISA formulation. Evidence: `build/exact-slide-probe/`.

### 11.5 Host sensitivity: faster rounding coverage

Evidence: `build/sensitivity-20261105/` (`sensitivity_report.md`, control hashes, patches, scripts, command manifests and partial results). All work was isolated; no per-command rounding gate was added to production.

Controls reproduced emulator smoke, mixed smoke/random, and recovered native hardware-oracle smoke/random hashes. The reverse experiment starts from the passing mixed mode and enables native u8 rounding for one entire shape+variant group. Grouping avoids arbitrarily splitting tiled work, although equal shape is not proof of exact logical-layer/Siamese pairing.

**16 variant-33 groups cover 616/629 commands (98%); 15 groups fail the smoke f32 gate.** Examples:

| Input → output shape | Commands | f32 smoke MAE vs 2.50 simulator |
|---|---:|---:|
| 1×10×130×64 → 1×8×128×64 | 160 | 0.39789 |
| 1×10×258×32 → 1×8×256×32 | 120 | 0.0280501 |
| 1×10×34×128 → 1×8×32×128 | 85 | 0.23056 |
| 1×10×258×32 → 1×8×256×2 | 40 | 0.00000500431 (unchanged f32; separate u8 regression) |

Model smoke f32 MAE is 0.0000746203 against the original reference. The sole f32 passer changes **7,683/163,840 u8 outputs**, all by +1: u8 MAE/bias **0.0468933**, max 1, whereas Model is u8-exact on smoke. Only the first **5/27** cases were evaluated for this candidate; all five pass the f32 gate, but all five differ in u8, and u8 absolute bias passes Model on only 1/5. The follow-up was stopped because this small separate-output relaxation offers no demonstrated broad speed win. It must not be described as a full27 result. Thirteen variant-33 command indexes in eight small groups remain unscreened.

An earlier all-native→one-exact-group experiment worsened error (an 80-command variant-32 group raised smoke MAE ~1.478→1.722 and random ~0.0924→0.1107). Errors interact non-monotonically, so one-at-a-time results are not an additive sensitivity model. No broad native-u8 replacement meeting the accuracy criterion was found; this is not proof that no combination or calibrated alternative can work.

Static-Fi cache review: Fi depends on filter rows, weight sums and input zero-point. VTCM pool identity is not immutable provenance; the 80 dynamic-weight preparations depend on current inputs. Persistent reuse needs proof of static producers or reliable generation tracking, not just pointer identity. Hashing all weights still pays the read cost. No cache was implemented or benchmarked. The review's inherited ~1.53 ms setup estimate is from older evidence; current measured HMX setup is ~5.35 ms, and not all of it is Fi work.

### 11.6 Fresh Model measurements and QAIRT 2.25

Connected Legacy headset serial `2G0YC1ZF860CBG` was probed through the existing signed vendor route. Vendor library fingerprints matched the preserved set; `mrsystemservice` remained running. Fresh smoke/random runs returned zero RPC/status and deterministic repeated outputs, with warm calls **9.248 / 9.723 ms**. Smoke u8 matched the simulator; random u8 differed in **96,462/163,840** values, MAE **0.7269836**, max **13**. Random f32 MAE against 2.50 remained ~**0.03307085**. This confirms end-to-end behavior, not the per-convolution instruction rule. DSP version API value `0x6` is recorded as returned, not interpreted as a proven ISA revision.

The user disconnected/reconnected Model during probing; no Model reboot, service, property, firmware, or vendor-file changes were made. Private fixture files were staged only under `/data/local/tmp/`.

**Official SDK located, downloaded and exercised:** [QAIRT 2.25.0.240728 archive](https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.25.0.240728.zip). The older download product path is `qualcomm_neural_processing_sdk`, while its internal root is `qairt/2.25.0.240728`.

- ZIP bytes: **892,905,658**.
- SHA256: **`c7395bd25a99c20f6a84ec6ecdba21c0cd7ccc4cd2b9450956ea69fd95f4e8f8`** (locally calculated; not claimed to be a published vendor checksum).
- Download: `.work/qairt-2.25-download/v2.25.0.240728.zip`, relative to `depth-model/`.
- Extracted SDK root: `.work/qairt-2.25-download/qairt/2.25.0.240728`.
- Selectively extracted 379 entries / 338,848,905 bytes: headers, x86 Linux tools/runtime, Android libraries, V69 DSP libraries, root metadata. Not a full SDK installation.
- Runtime testing uses amd64 Docker on the Mac. `qnn-net-run` needed `libpython3.10`; isolated image `depth-model-qnn225-sim` adds it to the existing simulator image.
- Provenance: `.work/qairt-2.25-download/manifest.json` and `frame/build/model-rounding-20261105/qairt-v225-summary.json`.

Keep **serialized-model generation version, vendor runtime version, and host simulator version separate**: the original baseline records QNN 2.19 for the context; the worker found a 2.25 vendor runtime version; the original host reference uses QAIRT 2.50. Downloading 2.25 does not by itself establish binary equality with the vendor build or unlock unsigned DSP libraries.

Controlled Conv2d fixture:

- A V69-targeted single-convolution fixture built under SDK 2.50 and executed in its x86 simulator; raw-context Model creation failed.
- Rebuilt with **2.25 headers and runtime**: generated **17,632-byte** context and successfully executed the host fixture, producing 8,192 u8 outputs, all 252.
- On Model, the 2.25 fixture still returned `create rc=0, status=1, model=0`. No inference/rounding conclusion follows from that failure.
- FastRPC logs confirm the private `libQnnSystem.so` file was **opened**, not that it successfully loaded/executed. The SDK V69 library is unsigned. Do not claim that custom DSP module execution was demonstrated.
- The vendor raw-context route needs System-library handling absent from the working preserved ExecuTorch/ET12 route. Debug-level/log methods tried were unsupported (`0x3eb`). No supported standalone custom HMX ELF execution was established.
- A supported ET12 wrapper for the tiny graph remains a possible investigation, not a completed test. The version-mismatch hypothesis is insufficient on its own: matching 2.25 did not make raw creation succeed.

Files: `frame/build/model-rounding-20261105/README.md`, `model-primary-smoke-random.json` and raw case directories; `oneconv/` (2.50 fixtures), `oneconv-v225/` (2.25 source, context, simulator output, Model create log).

**Full-model simulator comparison, using identical smoke/random inputs:**

| Case / head | 2.25 vs 2.50 simulator | 2.25 vs Model |
|---|---|---|
| Smoke u8 | Identical | Identical |
| Smoke f32 | MAE/bias +0.0000746203; max 0.00374985; 4,829 values differ | **Identical** |
| Random u8 | Identical | MAE 0.7269836; max 13 |
| Random f32 | MAE/bias +0.0001253946; max 0.00624847; 8,094 values differ | MAE 0.03304761; max 5.70921 |

Both float inter-simulator p95 values are zero. Thus SDK version accounts for the smoke float discrepancy and a small systematic float-output difference, but **not the random u8 discrepancy**. This is not proof of where inside the float-output path that difference arises. It also does not establish whether Model's u8 convolutions use simulator rounding, midpoint rounding with compensation, or another runtime treatment.

Reproduction/evidence: `frame/build/model-rounding-20261105/simulator-v225/{Dockerfile,command-v225.txt,inputs.txt,run-v225.log,comparison.json,Result_0/,Result_1/}`. The general runner is `tools/run_qnn_sim.py`; the saved command selects the dependency-complete `depth-model-qnn225-sim` image.

**Correction to earlier research prose:** `HMX_NATIVE_ROUNDING.md` and `QNN_RESEARCH.md` previously inferred that Frame's midpoint rule explains Model's output differences / is “QNN behaviour.” The experiments do not support that inference. Treat the measured Frame/ISS rule and the measured Model network outputs as separate facts until an isolated Model convolution test succeeds.

### 11.7 What the 4× target requires (model, not a promised result)

The user requested approximately **4× from the 42 ms accurate build**, i.e. **10.5 ms**. The existing run-list schedule model was retargeted to the final measured handler profile. It predicts **37.962 ms** scheduling versus **41.996 ms** measured DSP inference. The **4.034 ms gap** is not a proven irreducible overhead: it includes unmodeled work and model error. Measured DSP read/write work explains about 1.932 ms, but a complete attribution was not performed.

| Hypothetical change | Modeled schedule | With current 4.034 ms gap unchanged |
|---|---:|---:|
| Baseline | 37.962 | 41.996 ms |
| Remove modeled per-command dispatch overhead | 37.669 | 41.703 ms |
| All conv handlers 2× faster | 23.331 | 27.364 ms |
| All conv handlers 4× faster | 16.607 | 20.641 ms |
| Convolution free | 12.146 | 16.180 ms |
| Conv 4× and worker handlers 2× faster | 12.313 | 16.347 ms |
| Conv 8× and worker handlers 2× faster | 8.907 | 12.941 ms |
| Conv 6× and worker handlers 3× faster | 8.712 | 12.746 ms |
| All handlers 4× faster | 9.711 | 13.745 ms |

These use per-handler averages and fixed dependencies; they are projections, not measured lower bounds for every possible rewrite. Changed contention, batching, dependencies, data layouts, or approximation can invalidate them. Under this model, conv 6× + worker 3× plus reducing the remaining gap to 1.5 ms would be about **10.2 ms**. No implementation achieving those component targets has been demonstrated.

Artifact: `frame/build/ab/target-4x-20261105/{model.py,model.json}`. The script still depends on the earlier run-list decoder/source under `/tmp/frame-i-merge` and its old report for initial module loading, then replaces timings with the current report. Preserve or reconstruct those prerequisites before re-running; the saved JSON records the actual result.

**Supported direction, not a guarantee:** eliminate substantial expanded-input staging and repeated per-command setup/output work; reduce cost-volume preparation/layout work through proved reuse or fusion; reduce persistent invocation copies/overhead. An expand-once strip layout or a producer layout directly consumable by HMX is an architectural candidate, not a validated implementation. Reusing dynamic weights/scales across planes without dataflow proof is unsafe. Neither routine dispatch tuning nor a convolution-only improvement explains a credible 4× result on its own.

### 11.8 Remaining unknowns and handoff boundaries

- Exact Model convolution rounding/Fi construction remains unresolved; no tiny graph has executed there yet.
- No 10.5 ms candidate, no proved 4× path, and no new real-scene accuracy evidence.
- No broadly native rounding subset passed the tested gates without the documented separate-output regression.
- Fused 3×3 staging is rejected; do not merge it or replace the validated source/binary.
- No persistent Fi cache was implemented, and no safe data-generation scheme was established.
- The executor correctness issues in section 9 were not fixed in this pass; alternate-model validation was not extended for the new combined candidate.
- Luna workers handled Model probing, host sensitivity, and Frame code/tests at the user's request. The Model worker reached a usage limit; the primary agent completed the 2.25 download, matching-version host fixture, Model create attempt, and full-model simulator comparison directly.
- Working trees remain dirty with pre-existing and session changes. Saved source snapshots/patches and measured artifacts, rather than an assumed commit state or `/tmp` availability, are the reproducible record.

## 12. Exact planes and plan replay — 2026-11-05 (current status)

**Best validated accurate primary candidate: 14.598 ms warm median, down from 42.014 ms (2.88×).** With the optional `QDS_L2_NO_VERIFY` it is 14.284 ms (2.94×). Outputs are byte-identical to the mixed-rounding host oracle on all 27 synthetic cases, both heads, 2 runs each. Accuracy against the Model gate therefore carries over from section 11.1 unchanged: f32 MAE, absolute bias and p95 at or below Model on all 27, max on 22 of 27. Still no real-stereo validation.

The profiling-off rows of 12.2 say so; all other rows include its overhead (about 0.47 ms in the pairs).

### 12.1 What was found: exact u8 at the native rate

Section 6 and `HMX_NATIVE_ROUNDING.md` concluded that native-rate `cvt.ub` cannot be exact. That is wrong. It comes from QNN's `hmx_v75_convbbh1x1_stride1`, verified on the V75 ISS and then on Frame silicon. The `cvt.ub = acc(Rs)` selector has fields that the earlier probes never varied:

| Rs bits | Meaning |
|---|---|
| bit 0 | keep the accumulator (no clear or slide) |
| bit 1 | wrap modulo 256 instead of clipping |
| bits 12..13 | bias bank, loaded with `bias = mxmem2(Rt\|bank)` |

A scale word without bit 22 also floors instead of rounding. One native 64-pixel accumulator therefore yields q = floor(x / 2^pre) exactly, as two byte planes:

- Bank 0, scale `(24-pre)<<10`, selector 3: q mod 256.
- Bank 1, scale `(16-pre)<<10`, bias + 2^(12+pre), selector 0x1000: floor(q/256) + 16, clipped.
- HVX: `v = hi<<8 | lo`, clamp to 4096..8192, subtract 4096, then `vmpy_s1_rnd_sat(2q, M)`, which equals `(q*M + 8192) >> 14` exactly.

This needs no halfword A expansion and works with the staging-free 3×3 slide. All 926 u8 convs take it in the host model. Requires pre ≤ 7; anything else falls back to the wide path.

**Slide hazard:** the split store (`Rs|0xe0`, `Rt 0x700`) combines consecutive conversions, so alternating planes mix. Whole tiles are stored instead and the merge realigns one pixel: tile c pixel k is output column 8(c-1)+k-1. Reading input blocks in place still hangs silicon (see `HANDOFF_SESSION_20261102D.md`), so the window stays staged.

### 12.2 Measured progress

Warm median over the 27 cases unless marked. Every row marked 54/54 is byte-exact with zero worker exit errors.

| Change | Warm median | Gate |
|---|---:|---|
| Starting point (section 11) | 42.014 ms | 54/54 |
| Planes on the slide and native paths | 26.844 ms | 54/54 |
| Planes for `oh%8` convs (full tile computed, valid rows written) | 26.28 ms | smoke only |
| `QDS_POOL1_NOCOPY` (saves the 1.74 ms VTCM copy) | 24.440 ms | 54/54 |
| Compact stage window: fallback commands now fit the workspace; 32-bit block lookups | 22.137 ms | 54/54 |
| L1 flush limited to pools 3, 4, 6 and I/O (was 524,288 `dccleaninva` per run) | 21.357 ms | 54/54 |
| Per-command plan replay for slide convs | 19.754 ms | 54/54 |
| Parallel stage (≥16 groups), plan replay for the non-slide u8 path | 18.789 ms | 54/54 |
| *Frame rebooted; see the note below* | | |
| Plan replay for u16 convs (6/35, 6/34) | 18.748 ms | 54/54 |
| 1×1 convs read input in place (`ONE_INPLACE`) | 17.953 ms | 54/54 |
| Profiling off, TCM check on (`eng-prod.elf`) | 17.488 ms | 54/54 |
| *Back to profiling on for the rows below* | | |
| Single 16 MiB VTCM page; slide reads input in place; records built with HVX, output tables read in the merge parts | 16.87 ms | smoke only |
| Stride 2 (6/36) slide on the planes, in place | 16.190 ms | 54/54 |
| Setup fast paths (`SETUP_FAST`) | 15.609 ms | 54/54 |
| `call_args` and pool-array fast paths | about 15.45 ms | smoke pairs only |
| Dynamic-chain scale and inverse-scale without software divides (`CHAIN_LUT`) | about 15.15 ms | smoke pairs only |
| Profiling off (`eng-prod2.elf`) | 14.823 ms | 54/54 |
| Elementwise descriptor memo (`EW_MEMO`), and no software modulo per block in the requant/add, linearclip and data-mover loops | about 0.27 ms in pairs | smoke pairs only |
| **Profiling off (recommended build, `eng-prod3.elf`)** | **14.598 ms** | **54/54** |
| Optional: also `QDS_L2_NO_VERIFY` (`eng-prod3nv.elf`) | 14.284 ms | 54/54 |

**Compare across the reboot only through the pairs.** Right after the reboot XRService used a full core and the unchanged `np2` engine read 19.3–19.55 ms against 18.72 ms before it. It settled later, but `np2` was not re-measured then, so the 18.789 → 18.748 row understates the u16 gain. Alternating pairs in matching conditions: u16 plans 19.32 → 18.67 ms; in-place 18.13 → 17.98 ms; profiling off 17.96 → 17.47 ms; no-verify 17.96 → 17.52 ms.

`QDS_L2_NO_VERIFY` stops snapshotting the live lower half of L2-TCM and stops comparing TCM with its backup across the lease mode switches. The upper half is still saved and restored. It removes a check, not a function, so a TCM corruption across the switch would no longer be reported. Off by default.

`QDS_POOL1_NOCOPY` carries the previously known short-interval pool-1 difference (section 9). The full-model gates above do pass with it.

**Plan replay:** a conv's geometry and block tables are fixed for a bound program and pool table, so the first full run captures a small plan and later runs skip the executor, `setup()` and the checks. Weights and filter rows are still read every run, and out-of-range parameters fall back to the normal path. Plans are invalidated if the program, pools or conv registration change.

### 12.2b Why in-place HMX reads hung, and what QNN does

The firmware maps VTCM as two 4 MiB TLB pages. An HMX window walk whose blocks cross that boundary wedges the CDSP. Evidence: reading in place worked for the 112 commands whose windows stay in one page, with large and negative deltas; about half of all window groups cross, including the original hanging command (cmd 171, 32 of 36 groups); and once VTCM was remapped as one 16 MiB page for the lease (`QDS_VTCM_SINGLE_PAGE`, identity mapped, TLB restored before the lease ends), every window read in place, exact on all 27 cases. The crossing itself was never reproduced as a hang in isolation, so the cause is strongly supported, not directly proven.

QNN requests VTCM through `compute_resource_attr_set_vtcm_param_v2`, whose SDK header documents a minimum page size and "least page mappings". QNN's argument values were not read at run time, so this is consistent with its using one large page, not proof.

`CHAIN_LUT`: `s = max(a/16256, 1)` with 12 mantissa bits rounded has only about 2,065 values. The integer form `n = (32d+127)/254` was checked against IEEE float32 for every a up to 32640 and is wrong for a ≥ 32512 (s crosses 2.0), so those keep the float path. Inverse scales are a table filled by the original float code.

### 12.3 Rejected or no gain

| Experiment | Result |
|---|---|
| `-ffast-math` | 24.59 vs 24.43 ms: no gain, dropped, so the build has no relaxed arithmetic |
| Serial merge below 8 / 16 blocks | 24.50 / 24.88 vs 24.43 ms: worse |
| Serial stage threshold 8 / 32 groups | 19.42 / 19.42 vs 19.32 ms at 16 (smoke) |
| Dropping the slide's 2048-aligned input requirement | No change (26.96 vs 26.90 ms): those commands were already on the slide |
| QNN-style `dcfetch` of the next command's memos and tables | 16.27 vs 16.20 ms: handler time fell, but the prefetch cost the same. Removed |
| Single-load u16 3×3 walk | No Rs/Rt setting found on the ISS (131,072 combinations): the form needs a different load sequence. Not pursued |

### 12.4 Where the time is now

From the schedule model on the 16.190 ms report (`tools/crit.py`), a model and not a measurement, before the setup, table and modulo fixes: modeled 14.80 ms, of which 5.62 ms on the main list and 9.18 ms on the worker list, with 107 list switches. **The worker list is now the larger part of the critical path.**

| On the critical path | ms | µs per command |
|---|---:|---:|
| 6/33 u8 3×3 | 3.20 | 5.9 |
| 6/32 u8 1×1 | 0.66 | 6.2 |
| 6/35 u16 3×3 | 0.56 | 12.0 |
| 6/36 u8 stride 2 | 0.39 | 7.2 |
| Worker ops | 9.18 | long tail, mostly 2–7 each |

In the same model, making all worker ops 2× faster saves 3.8 ms and all convs 2× faster saves 2.1 ms. Setup now costs about 0.3–0.6 µs per command against 3–5 µs for kernels, so what remains is mostly kernel data movement.

### 12.5 Open work

- **Structural next step, not started:** the worker list runs serially on one thread while two helpers mostly wait for split jobs. The 80 dynamic-weight plane chains look independent of each other, and could run concurrently given real data-dependency analysis and per-lane memos and scratch. Not modeled, so the gain is unknown.
- A u16 replay bug wedged the DSP once (CTRL `(1,1,1)`, reboot needed, user approved). The plan kept `J.pool = g->pool`, a pointer into the stack `G`; replacing it removed the hang. A plan must keep nothing pointing at the stack or at near memory.
- `ONE_INPLACE` shows that in-place reads do not hang for single-tile 1×1 loads, narrowing the older "in place hangs silicon" note. In-place reads through the slide's `Rt` deltas were not retried and remain rejected.
- Ideas not started: a u16 slide for 6/35 (needs QNN's real load parameters, for example logged from the x86 simulator), epilogue fusion of conv + requant/add across the two lists, the remaining worker kernels, the remaining ~0.6 ms at lease end.
- Model experiments ran later, once the headset connected: see section 13.3. The isolated conv through the ET12 route was not attempted.
- The candidate source still contains diagnostic flags (`SLIDE_GEOM`, `SLIDE_PROF`, `EW_SHAPES`) that should be removed before merging.
- The first inference of a session is about 100 ms: only warm runs reach these numbers.
- The alternate model has no captured run list, so it gets neither run lists nor plan replay, and it was not tested on these builds. `QDS_POOL1_NOCOPY` was host-checked for primary only.
- Host coverage: the host model exercises the non-slide planes path. Slide and all plan replay exist only on Hexagon, so the Frame hash gate is their only full-model check. The slide kernel also has the randomized ISS test (`slidep_test.c`).

### 12.6 Reproduction and provenance

The candidate source is still in `/tmp/frame-speed-runlist/depth-model`, **not merged into the main tree and not committed**. Everything needed is in `build/ab/planes-20261105/`:

- `source-prod3.tar.gz`: sources of the current builds. `eng-prod3.elf` SHA-256 `668b47c851b33b1d48f5d91fb57dcc129a17db6737ee4fe2b3806123745a6c34`; `eng-prod3nv.elf` `13ca45846590e6d8bde92eec025ef861304ba48346c5f548525ecc7b44fb3a61`. Earlier stages have their own `source-*.tar.gz`.
- `*-all27.json`: gate reports. `build-*.log`: build logs. Earlier stages have their own `source-*.tar.gz`.
- `tools/planes_build.sh`: exact build flags, on top of section 11.2's: `-DNPRG2_PLANES=1 -DQDS_POOL1_NOCOPY=1 -DONE_INPLACE=1 -DQDS_VTCM_SINGLE_PAGE=1 -DSLIDE_INPLACE_ALL=1 -DSETUP_FAST=1 -DCHAIN_LUT=1 -DEW_MEMO=1`, with `PROFILE=0` in the environment (add `-DQDS_L2_NO_VERIFY=1` for the optional build). `tools/run.sh`: device gate wrapper. `tools/sens.py`, `tools/crit.py`: schedule analysis; these still need `/tmp/frame-i-merge` for the run-list decoder.
- `cvt_mode_sim.c`, `cvt_wrap_sim.c`, `two_cvt_sim.c`, `slidep_test.c`, `slide16_*_sim.c`: ISS probes behind 12.1.

`ab.sh` still labels these builds FAILED because it expects the emulator oracle. Use `tools/run.sh`, which checks the mixed host hashes.

## 13. Worker lanes from a dependency graph — 2026-11-05 (current status, supersedes the headline of section 12)

**First gated lanes build: `eng-l3.elf`, 12.672 ms warm median, down from 14.598 ms and 42.014 ms originally (3.32×); see the update below for the current best.** 54 of 54 runs byte-exact on all 27 cases, no worker exit errors. Accuracy against Model carries over from section 11.1 unchanged, because the outputs are identical. Still no real-stereo validation, and 12.672 ms is still above the 10.5 ms target and Model's 9.7 ms.

**Current best: `eng-l8.elf`, 9.681 ms warm median (4.34× from 42.014 ms), 54/54 byte-exact on all 27 cases, no worker exit errors; jitter `l8j` 54/54; adversarial serial orders `advl` and `advr` 54/54 each.** Past the 10.5 ms target. Roughly level with Model's 9.71 ms host time, and above its 8.8 ms DSP time; the Frame figure is DSP time. Still no real-stereo validation. See `MEMORY_LAYOUT.md`.

**Earlier stage: `eng-r0.elf`, 10.949 ms (3.84×), 54/54; jitter `r0j` 54/54.** See 13.5e.

**Earlier stage: `eng-l7a.elf`, 11.438 ms (3.67×), 54/54; jitter `l7aj` 14.151 ms, 54/54.** Smoke pairs of the steps after `l5c`, 3 rounds of 6 runs each:

| Build | Adds | Warm median, ms |
|---|---|---:|
| `l5c` | See the next table | 11.87–11.92 |
| `l6a` | Main runs ready worker commands up to 6 µs when it has no conv ready | 11.70–11.72 |
| `l6b` | `SLIDE_TRIM`: no unused stage records on the in-place path, no per-command deadline | 11.57–11.64 |
| `l6c` | `WEIGHTS_MEMO`: weight copies skip revalidation | 11.55–11.57 |
| `l6d` | `WEIGHTS_CPU_BELOW=2048`: tiny copies skip DMA | 11.48–11.50 |
| `l6e` | Those tiny copies move off main | 11.41–11.45 |
| `l7a` | Schedule regenerated from re-measured costs | 11.37–11.47; no clear gain over `l6e` |

Profiled accounting on `l5c`, 12.2 ms with profiling on: main busy 7.49 (almost all convs), main waiting on workers 3.48, lease end 0.44, lease begin 0.12, other time outside the lists about 0.2; worker lanes about 50% busy. Tested and dropped: a randomized priority search found nothing better than greedy in the model, and caching the conv zero-point sum is moot because every conv's input zero point is 0.

Earlier update, `eng-l5c.elf`, 11.957 ms warm median (3.51×), 54/54, jitter `l5cj` also 54/54. Its jitter build, `l5cj`, is 14.78 ms and also 54/54. It adds measured-cost scheduling, the signature check, the cheaper merge and free main order. Smoke pairs, 3 rounds of 6 runs each, in the order the changes stack:

| Build | Adds | Warm median, ms |
|---|---|---:|
| `l4a` | Measured costs, no-ops dropped, Fill and input off main | 12.44–12.51 |
| `l4b` | `QDS_L2_SIG` | 12.24–12.31 |
| `l5a` | `PLANE_SAT` | 12.00–12.08 |
| `l5c` | Free main order | 11.89–11.95 |

The Frame's outage was an orderly shutdown at 16:05 with a flat battery, not a DSP hang. The first `PLANE_SAT` device run failed on warm runs only: `plan_fi` was a second copy of the table builder and still used the old bias. Plan replay exists only on Hexagon, so host and ISS checks couldn't see it. Fixed; the signature check's error path was not exercised on silicon.

### 13.1 What changed

QNN keeps all HVX work on one serial worker list. Its list order costs far more than the real data dependencies require. `QDS_QNN_LANES=1` keeps the main list and splits the worker list over HT1 and HT2. Each list publishes a progress count, and each entry waits for the counts it depends on. HT5 stays a pure helper, and any list thread that is waiting serves parts.

| Build | Result | Coverage |
|---|---:|---|
| `prod3`, two lists | 14.50–14.54 ms | 3 smoke pairs, alternating with `l3` |
| `l3`, three lanes, schedule from handler averages | 12.63–12.67 ms | same pairs |
| `l3` | **12.672 ms** | **54/54 on 27 cases** |
| `l3j`, same schedule with random pauses injected per command | 16.106 ms | 54/54 on 27 cases, about 3.4 ms of added delay |
| `l4a`, schedule from measured per-command costs, no-ops dropped, Fill and input moved off main, DMA landing edges | 12.45–12.48 ms | 2 smoke pairs only |

Schedule model with measured costs, against about 12.0 ms measured main wall with profiling (a model, not a measurement):

| Schedule | Modeled ms |
|---|---:|
| Main + 1 worker, reordered | 12.71 (handler-average costs) |
| Main + 2 workers | 11.20 with no-ops dropped |
| ... with Fill and input moved off main | 11.02 |
| ... with main free to reorder (DMA queue order kept) | 10.86 |
| Main + 3 workers (would take HT5 away from helping) | about 0.5 lower, not pursued |

### 13.2 How dependencies were established

Declared tensors alone are wrong, and host checks caught that before silicon:

- **Spill, Fill and Zap declare no tensors.** 121 commands keep their memory in vector words. Their footprints are now decoded.
- **2 KiB blocks falsely share pool 3 tensors.** Linking a reader only to a block's last writer then loses real writer-to-reader edges once QNN reverses a false pair. The graph now uses 128-byte granules and all conflicting pairs.
- **Each conflict keeps QNN's guaranteed order.** Conflicts QNN leaves unordered (always main against worker) stay unordered, because they already race in QNN and in the two-list engine.

Checks, all on the host:

| Check | Result |
|---|---|
| Poison test: rerun each command with everything outside its declared footprint flipped | 0 hidden reads at 128 bytes, on smoke; control with declared reads removed from 200 commands flagged 117 |
| Pool diff after each command | 1,077,953 written granules, 0 outside declared footprints, on smoke |
| Adversarial topological replays (schedule, latest, random, main-first, worker-first) | 15 of 15 byte-exact on smoke, random and `h+8` |

**Limits:** the host runs the non-slide kernels, so these checks say nothing about the slide and replay kernels' exact footprints; only the Frame gate and the jitter build cover them. Host DMA is mocked, so async weight copies landing at their later checkpoint wait are only exercised on the device. The generator adds wait-to-consumer edges where QNN orders them; `l3` predates that edge and `l4a` has it. Races that neither the gate nor jitter triggered can't be ruled out.

`QDS_L2_SIG` (in `l4b` and `l4c`, untested on silicon) replaces the lease check's comparison against the uncached backup with a 64-bit signature of live TCM taken before and after each mode switch, so the lower half never goes through DDR. On the ISS it caught all 540 test corruptions, including paired flips and swapped vectors. Unlike `QDS_L2_NO_VERIFY` it keeps a check, but a signature can in principle miss a corruption that a full compare would catch. Expected saving about 0.3 ms, not measured.

### 13.3 Model measurements

| Item | Result |
|---|---|
| Warm host time, smoke | 9.25 ms median over 20 calls |
| `DELEGATE_CALL`, vendor profile mode 2 | 8.8 ms median (8.73–9.22) over 20 calls; that mode returns status 1 with intact outputs and gives no per-node data |
| Null RPC | 0.067 ms |
| Hardware threads reported | 6 |
| Node cycles | 4.65 M in total, of which 47% outside convs. These can't be pcycles (about 23 G MACs), so they are only relative |

**Model's u8 convs aren't using a cheap rule this engine could copy.** Against Model's recorded outputs on the host, the simulator's u8 rule is byte-identical on smoke and `flat_h0_id`. Silicon midpoint with a floor scale word gives smoke f32 MAE 3.26 and midpoint with rounding 1.48, against Model's 0.0001; both are rejected. Model's differences on textured inputs (58.9% of u8 on random) have another, still unidentified source. `hexagon_nnFetchIntermediateOutput` and the `Run4` intermediate path exist but look delegate-level, and were not called. The isolated conv was not attempted.

### 13.4 Rejected

| Experiment | Result |
|---|---|
| Offload conv merges from main, model only | 10.2 ms only if a merge costs the same on one thread as today split over about 3 parts; 12–13 ms at a realistic 2–2.5×. Not built |
| Enforce conflicts QNN leaves unordered (`--strict`), model only, at 2 KiB granularity with handler-average costs | About 1.9 ms slower, with no evidence it is needed |

### 13.5 Next, in order

1. Done: see the current best above. The outage was a flat battery.
2. Critical path after that is mostly convs (about 6 of 10.2 ms), with the worker tail next.
3. Kernel-level work per command now has real costs: `tools/lanes/cmdcost.json`, from `QDS_LANE_CMDPROF`.

### 13.5b Offline analysis while the Frame was down

**Instruction sets.** The SDK's `hvx_hexagon_protos.h` guards show no new HVX intrinsics in V75: they step 69, 73, 79, 81, and V73 adds only bfloat16 operations, which an integer model can't use. Every HMX form in QNN's six `hmx_v75_*` leaves also appears in its `hmx_v73_*` leaves. So V75 offers new routines, not new instructions, as far as QNN's usage shows. That is not proof about silicon: the `cvt.ub` selector fields of section 12.1 weren't documented either. QNN's `_sparsity` leaves were not studied.

**HMX clock.** `hmx_start` in `qd_hmx_power.h` sets 979.2 MHz, the top shipped point. The call path from the prepared engine wasn't traced, but the measured MAC rate rules out the 19.2 MHz power-on clock.

**HMX time.** On the slide path the wait for HMX completion is charged to the `mac` phase, so about 1.6 ms really is HMX and the 2.3 ms `output` phase is pure HVX merge.

**`PLANE_SAT` (built, not yet run on the Frame).** The high plane uses offset 239 instead of 16, so it clips exactly at q >= 4096. The merge then needs one saturating subtract instead of max, min and subtract, and the multiplier is stored doubled. That is 6 HVX ops instead of 12 per output vector. Exact on 1.4 M lanes (`bench/lanes/plane_sat_sim.c`), on three randomized HMX geometries on the ISS for slide and native paths (`bench/lanes/slidep_plane_sat.c`), and on six full-model host cases. ISS merge cycles fall about 17%, because loads and stores dominate; the ISS isn't a timing oracle.

**More Model rule tests.** Applying silicon midpoint or floor to only the dynamic-weight convs, or only to one conv class, moves output further from Model in every case. The u16 rule makes no difference to these comparisons. Rounding is not the source of Model's differences on textured inputs.

**Sensitivity, model only**, from 11.015 ms (main + 2 workers, Fill and input off main):

| What if | Modeled saving, ms |
|---|---:|
| All worker kernels 2× faster | 1.34 |
| All u8 convs 25% faster | 1.13 |
| 6/33 alone 25% faster | 0.81 |
| Copies 16/41, 42, 43, 45, 49 cost nothing (upper bound; some change format and can't be free) | 0.63 |
| Dynamic-weight chain 2× faster | 0.44 |
| u16 convs 2× faster | 0.35 |
| Transposes 2× faster | 0.30 |
| Weight copies 4/28, 29, 30 cost nothing (upper bound) | 0.29 |
| Add, resize or S2D 2× faster | about 0.11 each |
| **Requant 8/70 costs nothing (the fusion idea)** | **0.06: not worth doing** |

### 13.5c Arbitrary graphs on Model silicon

**Any QNN graph can now run on Model through the signed route.** Build a context with SDK 2.25 for V69, using the graph name `zelda_qvga_boltnn_delegate - delegate_0`, tensor IDs x, y, hidden, t = 1..4 and two outputs = 5, 6, in the model's NCHW shapes. `wrap.py` swaps it into the vendor's `dsp.pte` and fixes the flatbuffer offsets. Tools and logs are in `build/model-lanes-20261105/et-route/`. No vendor files, services or properties were changed; files stay under `/data/local/tmp/qd-qnn-20261030/oneconv-et`.

**Model's Conv, Add and Convert were byte-identical to the 2.25 x86 simulator in every graph tried.** 491,520 values at three multipliers for a 1→2 channel conv, then chains of 32- and 64-channel 3×3 convs and chains of Add and Convert at two sizes, with up to 256 distinct output values. So Model's differences on textured inputs in the real model come from something these graphs didn't exercise. Dynamic-weight convs are the obvious untested candidate, but that is a guess. This also retires the inference in section 6 and `HMX_NATIVE_ROUNDING.md` that Model's convs use a different rule: it holds at most for op paths not tested here.

**Per-op timing, from the slope of chains of 0, 32 and 64 ops (host time, 3 rounds, noisy):**

| Op | Model | Frame, from `cmdcost.json` |
|---|---|---|
| 3×3 conv, 32 channels, 20 output blocks | 2.95 µs | 6/33: 5.75 µs at about 25 blocks, but its channel counts vary from 32 to 128, so not like for like |
| 3×3 conv, fit over 4 sizes | about 2.0 µs per conv + 0.09 µs per block | Setup + staging about 1.9 µs per command |
| 3×3 conv, 64 channels, 160 blocks | 23.1 µs, 8.2 T MAC/s | Whole-model average about 4.3 T MAC/s, including overhead on small commands |
| Add, 40 KB | 2.16 µs, 54 ns/KB (rounds 2.3, 2.06, 2.16) | 10/13: 4.99 µs at 44.8 KB, 112 ns/KB |
| Add, 160 KB | 3.1–11.6 µs: too noisy to use | |
| Convert, 40 KB | 1.34 µs in 2 of 3 rounds; the third was invalid | 8/70: 6.02 µs at 56 KB, 107 ns/KB. Not confirmed to be the same operation |

Transpose chains are invalid: the slope is about zero, almost certainly because QNN cancels back-to-back transposes at compile time. The conv overhead idea is not supported: Model pays about the same per conv as the Frame. Conv kernels look within 1–2×; Add is about 2× behind on a newer chip, and Frame workload differs in tensor shapes and thread contention.

### 13.5d Dynamic-weight op on Model, and a fair per-op comparison

The model's `boltnn_quantized_bmm_tensor` was rebuilt as a QNN `MatMul` with its real shapes and scales: `A [1,80,64,128] × B [1,80,128,64] → [1,80,64,64]`, inputs 0.0241, output 0.6905. The compiled context contains the model's own kernels (`convert_weights_to_signed.packed`, `combine_scales`, `conv_scale_from_weights`, `inconv_im2col`). A first attempt was discarded: its output tail saturated everything to one constant, so its zero differences meant nothing.

**Exactness:** 0 differences from the 2.25 simulator over 40,960 samples, 2 inputs, 2 channel selections. Output range was narrow (62–81, about 20 distinct values) and the inputs were synthetic random data, so this is a weaker test than the conv chains. It doesn't identify the source of Model's real-model differences, which is still unknown.

**Timing on Model:** about 0.51 ms for the whole op, from 1.827 ms with the MatMul against a 1.318 ms baseline. That is consistent with its profile entry of 230k cycles at about 450 cycles/µs. Other profile entries are not reliable: some ops with dozens of Frame commands show almost no cycles.

**Frame run fully serialized for a fair comparison** (`--serial`, `l7s-serial.json`, `QDS_LANE_CMDPROF`): each command waits for the previous, with helpers free. Exact on smoke and random, 17.7 ms in total with profiling on. Handler sum 15.81 ms against 18.66 ms in the full run, so about 15% of handler time in the full run is contention or fewer helpers.

| Op | Model alone | Frame alone | Frame in full run |
|---|---:|---:|---:|
| **bmm, 80 planes** | **about 0.51 ms** | **2.76 ms** | 3.10 ms |
| Add | 2.16 µs at 40 KB | 3.93 µs at 45 KB | 5.24 µs |
| Requant / Convert (not confirmed the same operation) | 1.34 µs at 40 KB | 4.64 µs at 56 KB | 6.32 µs |

Model's graph used synthetic weights, and QNN may run parts of it concurrently, as the Frame does with helpers. Both figures are for an op running alone with its threads free, so the 5.4× is a fair order of magnitude, not a precise ratio.

Frame bmm per plane, running alone, 34.5 µs over 23 commands, most on tiny data: `convert_weights` 4.0, `Transpose.2D` 3.8, `Reshape.hw` 3.0, `ForceFormat_Crouton` 2.7, conv 2.6, `conv_scale_from_weights` 2.4, two `flat_to_vtcm` 2.1, `combine_scales` 1.9, `ForceFormat_Flat_c2f` 1.9, `Reshape@FB` 1.8, `bias_scale_shuff` 1.7, `weights_to_vtcm` 1.6, `invscale` 1.5, `scale_convert` 1.2, `convert_bias` 1.1. This looks like per-command overhead, not arithmetic, though per-command overhead hasn't been separated from kernel time directly.

### 13.5e Merge into the main tree, and the fix batch

**Merged:** the lanes code is now in the main working tree, **uncommitted**. `kmod/*`, `bench/recover_resident_exit.py`, `HMX_NATIVE_ROUNDING.md` and `HANDOFF_SESSION_20261102H.md` kept main's newer versions. The default host test still matches the emulator and a plain 3-thread reference DSP build still compiles. The main-tree build `m0` gated at 11.445 ms, 54/54; jitter 54/54. Backup of the replaced files: `build/ab/planes-20261105/main-tree-before-l7a-merge.tar.gz`. `SLIDE_GEOM`, `SLIDE_PROF`, `EW_SHAPES` and the host rule-hypothesis flags were removed. `bench/lanes/build_best.sh NAME` builds the current recipe.

**Fix batch, smoke pairs of 3 rounds × 6 runs (2 rounds for the first group), each against the row or baseline shown:**

| Fix | Warm median, ms | Result |
|---|---:|---|
| `m0` baseline | 11.446 | |
| HMX stays enabled for the run; 4/29 copies off main, CPU copy | 11.365 | Kept |
| Pause 64 / **128** on idle and waiting threads | 11.253 / 11.180 | Kept, then 255 |
| Per-command memo, vector `Reshape.hw`, `flat_to_vtcm` memo, HVX `scale_convert` and `combine_scales`, direct `weight_scales`, first-word sentinel | 11.345 against 11.365 | Kept; only −0.02 |
| Against pause 128 at 11.188: pause 255 | 11.110 | Kept |
| No barrier in the poll path (`CLAIM_NOBAR`) | 11.127 | Kept |
| Two per-command barriers removed (`LANE_BAR`) | 11.079 | Kept |
| All of the above (`q0`) | 10.962 | |
| Publish progress only where a list waits (`LANE_PUBLISH`, 2,061 of 4,165 positions) | 10.926 | Kept |

Rejected, measured:

| Experiment | Result |
|---|---|
| Sparse L2 save/restore, skipping all-zero blocks | +0.17 ms: the upper half is mostly non-zero, so the scan is a net cost |
| Pause 8 on waiting threads | +0.09 ms |
| Pause 64 in part joins | +0.03 ms |
| Split thresholds raised to 32 K elements | +0.17 ms |
| Split thresholds lowered to 2 K / 4 K | No change (+0.00 / −0.02) |
| Four parts instead of three | +0.02 ms |
| Skipping acquire barriers when thresholds repeat | No change |
| Steal limit 3 or 12 µs instead of 6 | No clear change |
| Constant-output command memo, analysis only | Only 0.24 ms qualifies, mostly copies that must still run. Not built |

**Main lesson:** idle threads were slowing busy ones. Longer pauses and fewer barriers together gave most of this batch's gain, about 0.37 ms from `o0`-level to `q0`, more than all kernel fixes. The mechanism (pipeline slots, or barrier and shared-memory traffic) was inferred from the results, not measured directly. Kernel fixes on `bmm` gave far less than its modeled saving, and the setup/kernel split shows setup is small (0.45 ms over the handlers measured), so the earlier "mostly overhead" reading of `bmm` was wrong. Those commands are slow inside their kernels, or rarely critical.

**Verification:** `q1`, same code before removing the rejected options: 54/54 twice, 10.97 and 10.96 ms; jitter 255 54/54; jitter 31 54/54 twice, 270 runs. Final `r0`: 54/54, 10.949 ms; jitter 255 54/54. Barrier removal is the part most exposed to rare races, and those runs reduce that risk without excluding it. The HVX float multiply `Q6_Vsf_vmpy_VsfVsf` matched scalar IEEE on 6.4 M ISS samples (the qf32 path mismatched about 40%), and the gate covers its real inputs on silicon. It needs `-mhvx-ieee-fp`.

**Next:** about 0.45 ms short of 10.5 ms. Largest lead: a memory plan fitted to the schedule (13.5f). Then conv staging and setup (about 1.8 ms), a u16 3×3 walk from `hmx_v75_convhbh_stride1_aligned`, and copy elimination (up to 0.63 ms, modeled, upper bound).

### 13.5f Is the hardware or the path limiting?

**No misconfigured clock in what was read.** Registers read from inside a live run (`QDS_CLK_PROBE`, `bench/lanes/device_clkprobe.py`): HMX PLL multiplier 51, 979.2 MHz. AXI PLL multiplier 78, divide 2, about 749 MHz, with or without `QDS_AXIM_RUN`, which is why that flag gains nothing. The core clock's PLL was not read; 1843 MHz comes from earlier upcycle measurements. Raw reads: about 11 GB/s from the backup region mapped as uncached, about 84 GB/s from VTCM on one thread. The VTCM figure is one vector load about every 2.8 cycles. Model's clocks were not measured.

**Rejected, smoke pairs against 10.89 ms:** idle back-off with extra pauses, +0.25 ms, so waits are latency-sensitive. Separate L1 flush per buffer, +0.15 ms, lease end 0.49 against 0.40 ms.

**Critical path, model with measured costs:**

| Constraints | Critical path | Modeled end |
|---|---:|---:|
| Today: QNN order on every conflict, DMA queue order | 8.67 ms | 10.37 ms |
| True data flow only, one HMX list, DMA queue order, 2 workers | 5.04 ms | 7.49 ms |
| Same, 3 workers | 5.04 ms | 7.07 ms |

About 0.6 ms outside the lists comes on top. So memory reuse and ordering appear to account for most of the gap, and a plan free of them models at about 8.1 ms in total, below Model's 8.8 ms DSP time. **This is an upper bound on the gain:** the model ignores capacity and contention. VTCM is nearly full (7.5 MiB of tile blocks touched of 8 MiB, plus 3.25 MB spill traffic), so a new plan can't remove all reuse. Crouton tensors go through block tables, which suggests blocks could be reassigned offline to suit the schedule, but nothing has been built or tested. Direct tensors would need new offsets.

**Parking idle threads: not attempted.** With no spare wake-up mechanism proven safe, a lost `resume` would hang the DSP. Back-off shows waits are latency-sensitive, so it only helps if wake-up is nearly instant. Rough best case about 0.2 ms, an estimate.

### 13.6 Provenance

Source: the main working tree, **uncommitted** (merged from `/tmp/frame-speed-runlist/depth-model`, which is now stale). Snapshots in `build/ab/planes-20261105/`: `source-l3p.tar.gz` (the `l3` code), `source-l4.tar.gz` (later, including the signature check). The generated header `native_program/qnn_lanes_primary.generated.h` belongs to the last generator run, so regenerate it to rebuild a specific schedule.

- `eng-l3.elf` SHA-256 `3f9046a4e9fba5cb49e41fd6b7e7f0abb4ea079610329f62ef2e94e532efcccd`; gate `l3-all27.json`, jitter `l3j-all27.json`.
- Current best: `eng-l8.elf`; gate `l8-all27.json`, jitter `l8j-all27.json`, serial orders `advl-all27.json`, `advr-all27.json`; build with `bench/lanes/build_best.sh` after generating the lane header with the options in that script.
- Earlier: `eng-r0.elf` SHA-256 `c02034d6fc1c168376a7af2e48e8a306301a694dc10fdbc632d6e03d37c2d7de`; gate `r0-all27.json`, jitter `r0j-all27.json`; source `source-r0.tar.gz`.
- Earlier: `eng-l7a.elf` SHA-256 `f66a7c8e8b593f61088219ddc208e5099f3d23c3e2f0853ba1a8be1be70f1076`; gate `l7a-all27.json`, jitter `l7aj-all27.json`; source `source-l7a.tar.gz`. Generator flags: `--workers 2 --costs cmdcost2.json --drop 1/10 --move 26/8,3/1,4/28,4/27 --cpu-weights 4/28,4/27 --free-main --main-steal-us 6`. Build flags add `-DQDS_L2_SIG=1 -DPLANE_SAT=1 -DSLIDE_TRIM=1 -DWEIGHTS_MEMO=1 -DWEIGHTS_CPU_BELOW=2048`.
- Previous best: `eng-l5c.elf`, gate `l5c-all27.json`, jitter `l5cj-all27.json`, source `source-l5c.tar.gz`.
- `tools/lanes/`: generator, replay scripts, measured costs, device profiler. Generator flags for `l4a`: `--workers 2 --costs cmdcost.json --drop 1/10 --move 26/8,3/1`; add `--free-main` for `l4c`.
- `build/model-lanes-20261105/`: Model probe source and profile. `build/ab/planes-20261105/{base,floor,mid}.json`: rounding comparisons.
- Build flags: section 12.6, plus `-DQDS_QNN_LANES=1` (optional `-DQDS_L2_SIG=1`, `-DQDS_LANE_JITTER=255`, `-DQDS_LANE_CMDPROF=1`).
- `tools/run.sh` now deletes old reports first and fails if the Frame is unreachable. Before this fix it could print an earlier run's result as if it were new, which happened once during this work and was caught.
