# Padded direct 1x1 SDK numerical harness — 2026-10-30

Offline only: Qualcomm Hexagon SDK 6.6.0.0, Tools 19.0.07, `hexagon-sim -mv75 --quiet`, four real NativeGraph CPU-prefix `.ncpfx` fixtures. No Frame device was accessed. The harness command is `sh frame/native_program/run_prepared_native_padded_cpu_prefix_sim.sh`.

## Results

The numerical oracle is the fixture's NativeGraph CPU output over all bytes in the 32 physical output pages (65,536 bytes per case). Deltas are unsigned output-byte differences, native minus CPU.

| Fixture | command / sequence | NPRG refs X,W,Fi,Y (QDRX source IDs in parentheses) | raw/logical CI | staged packed bytes | Exact | +1 | -1 | Other |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| primary first12 | 69 / 115 | 88,67,69,112 (89,68,70,113) | 12 / 9 | 1,024 | 63,920 | 1,616 | 0 | 0 |
| primary CI20 | 3069 / 3416 | 2859,2775,2777,2879 (2860,2776,2778,2880) | 20 / 18 | 4,096 | 64,720 | 816 | 0 | 0 |
| alternate first12 | 63 / 108 | 88,55,57,100 (89,56,58,101) | 12 / 9 | 1,024 | 63,920 | 1,616 | 0 | 0 |
| alternate CI20 | 3020 / 3364 | 2879,2635,2632,2882 (2880,2636,2633,2883) | 20 / 18 | 4,096 | 64,256 | 1,280 | 0 | 0 |

All four commands passed the padded 1x1 preflight, submit/drain, packed-weight byte comparison, output-page comparison bounds, tail checks, and pool preservation checks. The output tables pointed only to fixture Y pages. Every byte in active VTCM outside those Y pages matched its pre-execution snapshot; every other nonempty pool matched its snapshot. The 128-byte `0xA5` packed-weight guard remained intact. Bytes after `tail_publish_bytes` through the reserved 512 KiB tail matched their snapshot. The only numerical differences were positive one-byte deltas; no negative or larger deltas occurred.

The first attempts exposed test harness configuration issues, not numerical-kernel failures: Hexagon `long` is 32-bit signed, so comparing the file size to `(long)UINT32_MAX` rejected all positive input sizes; and the mock VTCM simulator needs `QD_DIRECT_ALLOW_MOCK_BASE=1`. Both fixes are in the harness only. NPRG2 tensor IDs are lowered/reindexed from QDRX IDs; command index, sequence, opcode/variant, input slot roles, geometry and output pages identify the matching operation.

## Provenance hashes

Harness and fixture SHA-256:

- `test_prepared_native_padded_cpu_prefix_sim.c`: `6cfc90854a6e27fbc74a057626067b8c8aa5d385d26f59c1765274b1356c87bd`
- `run_prepared_native_padded_cpu_prefix_sim.sh`: `5d6a968c8226bc7c2d2b2f0b29737d3ab9a819bbcf56dfe28e4c077fe364909a`
- primary first12 fixture: `37f97237e1901f5e9418ab34da412b6adfcd54cd4ebd51c4717969aab09fee63`
- primary CI20 fixture: `e96bbc29de0afe8a6320a2c524eeabfb4b1bbc0c9ddda21bcf162228ad9566fe`
- alternate first12 fixture: `1b93ebfa02897e981d42ed0a99f18aaf44b5fa721e4ff2477c925932ef4167e8`
- alternate CI20 fixture: `b634f89e167a1a05ed971b16858123c32f5548b58724bd6813cd44d4a3a0798f`

Numerical implementation inputs:

- `prepared_native.c`: `a051338827ae2bee1e47b79370d5b21439b032d6b52f4ed39e63e9f1dbce2a87` (matches the handoff hash)
- `prepared_executor.c`: `be6056da6bfd212b0c58f72fb463c61ff8ff2361c7816aec42e61afc3ff7ee8d`
- `nprg2_runtime.c`: `9681ad2ef452c8c3b3e36cebdc9bb13c42cc8a9e83911d6bb1fdc59d358d6682`
- `qd_native_direct.c`: `f85e86dad8da31ad14749d0fd27d0a27af83470a2de7363964b972e2439cd5b9`
- HMX leaf/finish assembly: `qd_qnn_hmx_1x1.S` `4b65fe053f11fd243eaabc86b0bd5fb72c8003e76021587faf5096164e405142`; `qd_hmx_ops.S` `ae30962a6a11a93a51f33ccf11f74312d316f2e2e8f97a59865060092e5612f1`; stride1 setter `7b18d8a11bb0e7eb278ddc2d1590acf9356ae9498148beb4d6f621648a342bf6`.

The frozen padded engine remains a build-only artifact; this SDK proof does not validate or authorize its Frame execution.
