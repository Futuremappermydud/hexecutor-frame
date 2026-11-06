# QAIRT direct 3x3 convolution investigation

## Source provenance

- Provider: `/Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/libQnnHtpV75.so`
- Provider SHA256: `4282ff5880c0b0b609da31769a65a101f9a2beb6b536554cbb14753370c2ab1a`
- `hmx_v73_convbbb_stride1`: ELF VA `0x388cc0`, 996 bytes, SHA256 `7c5c198ea6b56c44fc129f9655154ba770f7131fd4ef7fa0feda170c9abf5ce2`
- `hmx_v73_convbbb_stride1_aligned`: ELF VA `0x3882c0`, 824 bytes, SHA256 `12fb83a271502763f9d0770fa4e0ecdefca4e9dc7c75817dcb4ba2d575788eb6`
- Disassembly generated offline with:
  `docker run --rm --platform linux/amd64 -v /Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned:/qnn:ro -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" depth-model-hmx-tools:latest /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump -d --start-address=0x3882c0 --stop-address=0x3885f8 /qnn/libQnnHtpV75.so`
- Captured output SHA256: `2b38f7eb1683f7a60810c6ac374641809c3212ce110d1869b7eee805fdef3c63`

## What disassembly establishes

This is the `bbb` (u8 activation, signed byte filter) stride-1 HMX entry family. The 996-byte wrapper reads the flags word at config `+0x30`; if bit 3 is clear it jumps directly to the aligned entry at `0x3882c0`. This only establishes branch selection, not that a specific caller satisfies every precondition.

Both entries use the six-argument ABI already observed in `qd_qnn_hmx_1x1`: `r0=output descriptor`, `r1=input descriptor`, `r2=filter bytes`, `r3=FI/bias bytes`, `r4=config`, `r5=mode`. In the aligned leaf, disassembly reads config words at `+0x30`, `+0x18`, `+0x08`, `+0x00`; input descriptor words at `+0x00,+0x04,+0x08,+0x0c,+0x10`; output descriptor words at `+0x00,+0x04,+0x08,+0x0c,+0x10,+0x14`. It uses descriptor word 0 as a pointer table and word 1 as a scalar. Word offsets alone do not identify tensor shape/strides; these fields are not yet fully decoded.

The aligned leaf has runtime branches on config `+0x30` (sign, bits 0 and 1), input descriptor dimensions/strides, and mode. No branch other than the wrapper's bit-3 dispatch is considered safe or covered here. Do not call this kernel based on shape alone.

The descriptor mapping proposed during this investigation is `+4` depth-block stride, `+8` crouton-row stride, `+0xc/+0x10` actual input H/W, and `+0x14` channel count. The smallest attempted spatial fixture used input `{table,1,2,10,10,32}` and output `{table,1,1,8,8,32}`, with a 2x2 input-crouton pointer table. Attempts 01–09 are retained as logs. Correcting the spatial shape avoided the original flattened-table mistake, but the leaf either exhausted the output table or failed to complete. The latest attempt used `cfg[2]=cfg[3]=0`, `cfg[6]=1023`, two output table entries, and 100,001 completion polls; it ended with `done=0xffffffff` and output poison unchanged. This is a failed attempt, not a supported configuration.

An offline trace of setter callsites found calls to `_Z22set_hmx_params_stride1P10hmx_paramsmmmmmmm`, but the inspected `0x7a76d4` site is an error/logging path, not a useful ConvLayer execution path. The leaf has no direct callxref in the provider; dispatch appears indirect. I did not recover a caller construction path for the spatial HMX cfg fields, so the spatial control words remain unknown. In particular, the known 1x1 constants `cfg[3]=2047` and `cfg[6]=1023` are not established for spatial.

## Graph inventory

`recovered/primary/graph.json` contains 956 serialized `ConvLayer_s1.opt` nodes. The frequent 3x3 u8 signatures include 160 nodes of 64→64, 120 nodes of 32→32, 85 nodes of 128→128, and 40 nodes of 64→32. There are also u16 3x3 nodes (including 32→32, 64→32, and 32→64). These are serialized graph nodes, not a count of independent runtime invocations. Actual eligibility must also match tensor format, padding/halo layout, weight packing, FI encoding, config flags, and supported descriptor branches.

## Validation status

No direct 3x3 simulator fixture is claimed. The known `qd_qnn_hmx_1x1` fixtures establish only the 1x1 leaf. The 3x3 ABI still needs a proven map from descriptor fields, HMX controls, and packed filter layout to the graph's blocked input/output representation before a fixture can be considered meaningful. Keep the new leaf out of execution until that map and output/guard checks are proven.

## Bounded 1x1 expansion and graph geometry

The new `qnn_hmx_1x1_graph_sim.c` fixture and independent checker exercise 14 cases across the seven eligible graph signatures (identity and random weights/inputs). They verify all bytes in the physical 8x8-crouton output allocation, including crop-external rows for logical H=2/4, table-tail poison, and output guards. Results are in `qnn-hmx-1x1-graph-sim.json`; this establishes simulator output/guard behavior for the fixture geometries only. It is not hardware validation and does not qualify every serialized node's runtime flags.

`report_1x1_graph_coverage.py` counts 140 of 280 serialized `1x1 ConvLayer_s1.opt` nodes as shape/dtype eligible for the proven u8 channel-multiple-of-32 leaf. Eligible channels are CI=CO=128 for these graph signatures; the separate synthetic expansion fixture also covers CI/CO multiples of 32 through 256, but is not included in this graph count. The report records raw tensor dimensions separately from the effective output dimensions after `k_hmx_conv` rounding/clamping and the physical crouton allocation. The 40 u16 nodes, 80 nodes with CI=9, and 20 nodes with CI=18 remain excluded.

Graph simulator fixture provenance:

- C source SHA256: `3affb235f7a3c1055d0485cc6a4d18ad68e214db7e7999ba4a9d569838549c7d`
- independent checker SHA256: `d58f7c374eb638ba7ddc30eb416b8f46cab1530342ad0ba6b12d13aabaaab75f`
- result JSON SHA256: `e1f84bfe78e445e9c6b16a5c1e7cbf96eb7463ef66e48708a88d7f6bb75b8bc7`
- simulator log SHA256: `a66941770435045beac7e4f9530e256a6ade1a66872da4b39cb09e66db650b87`
- leaf assembly SHA256: `5abacbdf331202c684a462d71cc344b8c21bb2cb2fbf2310086a68e46c7898e3`
- 3x3 fixture source SHA256: `674585d7640aa40e9f840f6b5422eee2231aa08c33b27bba477ea6176c7c3e04`

The simulator uses the local Hexagon SDK 6.6.0.0 (`hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2`) and `depth-model-hmx-tools:latest`; no device was accessed. Re-run the graph fixture from the repo root with:

```sh
docker run --rm --platform linux/amd64 -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -fno-stack-protector -Wall -Wextra native_kernels/qnn_hmx_1x1_graph_sim.c native_kernels/qd_qnn_hmx_1x1.S qd_hmx_ops_encoded.S -o native_kernels/qnn_hmx_1x1_graph_sim.elf
docker run --rm --platform linux/amd64 -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet native_kernels/qnn_hmx_1x1_graph_sim.elf
.venv-cpu/bin/python frame/native_kernels/check_qnn_hmx_1x1_graph_sim.py
.venv-cpu/bin/python frame/native_kernels/report_1x1_graph_coverage.py
```
