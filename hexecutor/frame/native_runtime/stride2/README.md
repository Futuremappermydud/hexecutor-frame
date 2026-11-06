# Legacy V69 raw stride-2 capture

This package is an offline SDK simulation probe for primary QDRX `ConvLayer_s2.opt@CB*2.Xwb.Fi` sequence 224. It does not modify shared runtime or existing adapters.

## Provider path and fixture

The primary QDRX command has exactly three inputs `[CB, Xwb, Fi]` and one output. The pinned V69 provider contains the legacy four-tensor RTTI signature `conv_layer_quant_stride2(QUint8Crouton_TCM, SWeights_TCM, Int32_TCM)` and a wrapper at VA `0x21bfd4`. It loads fields `+4,+8,+0xc,+0x10` into worker `r1,r2,r3,r0`; for this command these are input CB, Xwb weights, FI, and output. The worker calls `0x20c960`, then helper `0x20cb50`, and dispatches stride-2 leaves including `hmx_convbbb1xN_stride2` at `0x3c7d00`, `hmx_convbbbNx1_stride2` at `0x3c7f40`, and `hmx_convbbb_stride2` at `0x3c8a80`.

`extract_v69_s2_helper.py` extracts the exact worker, helper, leaf bodies, and setter span from `.work/model-qnn-20261030/libQnnHtpV69.so`. Source provider SHA256 is embedded in the generated assembly. The simulation builds with `build_v69_s2_capture.sh` and mounts `frame/native_stride2/s2_fixture_224.bin` at `/src/native_stride2`.

## Captured helper ABI

The first harness mislabeled the operation fields as output, input, weights, FI, rotating tensor roles and selecting an invalid 1xN route; those captures/runs are superseded. With corrected fields, fixture 224 selects the generic leaf and produces these arguments at entry when the provider call packet is preserved:

```text
r0 = stack request record (starts with output page-table pointer)
r1 = stack input descriptor (starts with input page-table pointer)
r2 = raw Xwb pointer (first generic-leaf weight load uses this address)
r3 = raw FI pointer (bias preload uses old r3 in its update packet)
r4 = stack HMX configuration
r5 = 128 input channels
```

The fixture tensors are X `[1,9,130,128]`, max X `[1,16,136,128]`, Y `[1,8,128,64]`, W `[3,3,32,64]`, and FI `[1,2,1,64]`. Quantization scales are X `0.04590979963541031`, W `0.009293106384575367`, and Y `0.07370951771736145`; all zero points are zero. V69 dtype IDs are U8 `1` and signed int8 `7`.

The 1xN branch call at `0x20cafc` has a delayed `r3:r2 = combine(r17,r16)` in the same call packet. The extractor preserves it in an explicit assembly packet. Captures taken through a C wrapper, or captures that omit the packet companion, report incorrect operands and are not valid ABI evidence.

## Current result

The corrected helper builds descriptors/config and the actual generic leaf completes its HMX drain in V75 SDK SIM. The numerical result is not a proof: fixture 224 differs from CPU output in 47,931 of 73,728 allocated bytes (266 `+1`, 232 `-1`, 47,433 other); the first 32 descriptor-active pages alone differ in 39,740 of 65,536 bytes. The remaining four allocated pages are outside the descriptor-active page count. The helper-built leaf descriptors and configuration are captured at entry. The unresolved mismatch blocks production integration; first establish exact input/output and raw-weight layout, then repeat fixtures 363 and 3407.
