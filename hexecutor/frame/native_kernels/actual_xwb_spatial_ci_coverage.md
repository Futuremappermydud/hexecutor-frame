# Actual 3x3 Xwb CI coverage on the generic V75 HMX leaf

This records CPU-prefix fixtures captured from the primary QDRX graph and executed by the generic V75 spatial leaf in the Hexagon simulator. These are simulator proofs, not hardware validation.

## Proven actual graph nodes

| Sequence | Kernel | Input logical / max | Input pad | Output logical / max | Output slice | Active output bytes | CPU comparison |
|---:|---|---|---|---|---:|---:|---|
| 247 | 3x3 Xwb, CI64, CO64 | 10x130x64 / 24x144x64 | 7,7 | 8x128x64 / 8x128x64 | 100 | 65,536 | 64,863 exact; 673 +1 |
| 874 | 3x3 Xwb, CI128, CO64 | 10x130x128 / 24x144x128 | 7,7 | 8x128x64 / 8x128x64 | 0 | 65,536 | 64,024 exact; 1,512 +1 |
| 3589 | 3x3 Xwb, CI128, CO128 | 10x66x128 / 24x80x128 | 7,7 | 8x64x128 / 8x64x128 | 44 | 65,536 | 64,154 exact; 1,382 +1 |
| 3527 | 3x3 Xwb, CI256, CO128 | 10x66x256 / 24x80x256 | 7,7 | 8x64x128 / 8x64x128 | 84 | 65,536 | 64,026 exact; 1,510 +1 |
| 5019 | 3x3 Xwb, CI32, physical CO32 / logical CO2 | 10x258x32 / 24x272x32 | 7,7 | 8x256x2 / 8x256x32 | 0 | 65,536 physical | 65,335 exact; 201 +1 |

For all five cases, the CPU-prefix fixture contains the physical input pages before the target command, raw Xwb weights and FI, and the physical Y block table after the CPU command. The simulator compares every physical output byte to that CPU output: only deltas 0 and +1 occur. Each run also checks input/output canaries and unused block-table entries. Staged X, W, FI and CPU-output hashes are independently checked against the fixture by the host runner. Sequence 5019 checks the full CO32 footprint, including the 30 aliased lanes beyond its logical CO2 output.

The CI64/CO64 case exercises a nonzero Y slice. Its active 32 pages (table indices 100–131) are normalized into simulator table slots 0–31, and the descriptor points to the aligned normalized table base. The CI256/CO128 case similarly normalizes indices 84–115. This mirrors the adapter's normalized active-table construction and avoids passing a pointer into the middle of the original table.

## Descriptor values and spatial leaf

The passed runs use these descriptor conventions:

- Input depth-block step: `CI / 32`.
- Input row-of-croutons stride: `ceil(input_max_width / 8) * (CI / 32)`.
- Input end coordinates: logical width/height plus their input padding, not max width/height. The values are 137x17 for sequences 247 and 874, and 73x17 for sequence 3527.
- Output depth-block step: `CO / 32`.
- Output row stride: `ceil(output_max_width / 8) * (CO / 32)`.
- Output end coordinates: logical width/height plus output padding.
- Setter arguments for these 3x3 nodes: `(cfg, 0x700, KH=3, ceil(input_max_height/8)=3, CI, 7, 1, 12)`; the resulting cfg words are `e0,700,140,71f,740,71f,17ff,bff,0,0,0,0,c`.
- Generic leaf mode is `{1,0}` for the graph's uint8 activation / signed int8 Xwb path.

Raw FI shape is `[1, CO/32, 1, 64]`: each output depth block contains 32 multipliers followed by 32 biases. The finish FI is initialized in full (64 words), with the first 32 conversion words set to `0x404000` and the remaining 32 set to zero. The adapter now copies complete output croutons with one aligned 2 KiB HVX copy and reduces its guard regions with HVX before scattering partial edge croutons lane by lane.

## Reproduction

From the repository root, run:

```sh
./.venv-cpu/bin/python frame/native_kernels/run_qd_native_3x3_adapter_sim.py
sh frame/native_kernels/run_qd_native_3x3_prepare_reject_sim.sh
./.venv-cpu/bin/python frame/native_kernels/check_qd_native_3x3_graph_eligibility.py
./.venv-cpu/bin/python frame/native_kernels/census_qd_native_3x3_graph_eligibility.py
sh frame/native_kernels/run_qd_native_3x3_graph_bridge_eligibility_sim.sh
./.venv-cpu/bin/python frame/native_kernels/run_qd_native_3x3_slice_candidate_sim.py
```

The adapter runner regenerates all five CPU-prefix fixtures, compiles with `depth-model-hmx-tools:latest` and SDK `6.6.0.0`, runs `hexagon-sim -mv75 --quiet`, validates the physical output and staged hashes, and writes the JSON report and HMX output binaries. The rejection proof tests wrong geometry, out-of-pool block references, and weight/workspace aliasing; each returns before any of the 1 MiB workspace changes. The graph-bridge gate test generates T/C metadata from the actual QDRX records and calls the actual source-included bridge: supported metadata reaches the deliberately disabled HMX ownership error, while an unsupported geometry defers. Cache operations, adapter calls, and all 8 MiB of fake VTCM remain untouched. No Frame or Model device was accessed.

## Artifacts and provenance

- Detailed adapter report: `qd_native_3x3_adapter_sim_report.json`; full graph metadata census: `qd_native_3x3_graph_eligibility_census.json`.
- CPU-prefix fixture SHAs: 247 `e7f6f22a833befb58114a4ead798bcd399811cdf6ec7c3164a38fb34f66ca132`; 874 `522a9c32712413ac23a0697c44220c5adca7fd235bea08cea628cdaef337c693`; 3589 `ef7b0dc04406f7d75c45b532a3912979628cbbb4c58bf4e4baa854cfbfb823cc`; 3527 `ec2697f0ed857476888c0edb96ac714a0703437e35ac5ea0aa23e1407d3e2d3d`; 5019 `19239c5af953fc6e9a2f884da428d6301f1d6cc0861254e628d1967ef3662921`.
- Odd-Y-slice candidate artifacts: sequence3627 fixture `448e8c3b03ce03608af9d5a72905a3b822b173169bf9357d346c888244566f74`; candidate C `bf5440146445a26a8cdcef86b5c129ae6e3f77427ef78b92d8d777281658428b`; report `4cbdf301f29a8ed9968713e76a370001abf7e14fccdd0964dca7d1c9f95ed6f0`.
- Shared generic leaf assembly SHA-256: `964ec477c48ce528e8dfef7cbd51925f8ebedc15af51e5c915e18f95de32cb05`.
- Setter assembly SHA-256: `7b18d8a11bb0e7eb278ddc2d1590acf9356ae9498148beb4d6f621648a342bf6`.
- Completion assembly SHA-256: `ae30962a6a11a93a51f33ccf11f74312d316f2e2e8f97a59865060092e5612f1`.
- Frozen adapter C/H and graph bridge SHA-256: C `7f6040e615e921ae64e25edbecb6c6d2af2d866bf20624a3853b0fc4f40cf743`; adapter header `c7e01788c8fa1cbc1fb76ab41daf98aec3b6b7dba45f09793a14db96376e24e7`; graph bridge `d9624e54a06c06ed25840d7d09efbd2b4e2a94c13dbc4c8e639aa2a554023cc7`.
- Fixture generators, simulator C sources, runners, and built simulator ELF hashes are in the per-sequence JSON reports and can be reproduced with the commands above.

The frozen graph bridge metadata/table gates admit 510 of 956 stride-1 convolution commands in the primary graph and 524 of 956 in the alternate graph. The 23 height-profile near misses in each graph are 13 `X4x10/max16x24 -> Y2x8/max8x8` cases and 10 `X6x18/max16x32 -> Y4x16/max8x16` cases. Another 73 primary and 59 alternate commands fail the conservative output block-slice alignment check. A separate simulator candidate removes only that output modulo check; census table/pool checks then admit 583 commands in each model, and actual sequence 3627 (CI128/CO128, Y page slice67, `67 % 4 = 3`) passes the full 65,536-byte physical CPU oracle at 64,457 exact and 1,079 +1. All model input slice offsets are zero, so this gives no evidence for removing the input modulo check. Neither that candidate nor its relaxed check is in the frozen bridge. The accepted current census has no output spatial crops; 40 commands use the separately proven logical CO2 / physical CO32 write footprint. Only the five table sequences and candidate sequence 3627 have individual simulator numerical proofs; the census does not establish numerical correctness for every admitted command, S2 kernels, other activation types, or live graph leases.
