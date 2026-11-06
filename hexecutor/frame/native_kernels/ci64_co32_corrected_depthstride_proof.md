# CI64/CO32 raw-Xwb generic leaf proof

This simulator fixture validates the generic V75 spatial leaf for one input depth-block lane group that the previous CI64 probes did not reach. The earlier CI64/CO32 files used descriptor word `+4 = 1`, copied from CI32, despite CI64 having two 32-channel input blocks per spatial crouton. The generic leaf uses this descriptor field as the depth-block step. With `+4 = 2`, the ordinary SDK Xwb byte order works; the earlier guessed alternate B packings were not evidence of a kernel or weight-layout limitation.

## Fixture and oracle

- Input: logical 10×130×64, with 7-pixel top/left origin in 18×3×2 input crouton blocks. Input table row stride is 36 blocks. Descriptor: `{table, 2, 36, 137, 17, 64}`.
- Output: logical 8×128×32 in 16 output croutons. Descriptor: `{table, 1, 16, 128, 8, 32}`.
- Filter: 3×3 Xwb, one signed-byte weight for each `CI=32..63 -> CO=0..31` pair at each stored tap; raw SDK Xwb layout `[O/32,I/32,H,W,(I%32)/4,O%32,I%4]`.
- Oracle: independently reads the physical crouton input location for each output at `(y+ky, x+2-kx, ci=co+32)`. The `2-kx` coordinate accounts for the Xwb width reversal used by stride-1 Xwb.
- Every stored tap (all nine) matches exactly over all `8×128×32 = 32,768` logical output bytes. Each call drains immediately. Input guard (12 KiB), output guard (2 KiB), and both 256-entry block-table tails pass.
- Config emitted by the setter: `e0,700,140,71f,740,71f,17ff,bff,0,0,0,0,c`. FI and finish zero-FI are fully initialized; zero-FI is 64 words with scale `0x404000` in words 0–31 and zero biases in words 32–63.

This demonstrates the generic HMX simulator path for CI64/CO32 with two input depth blocks and raw Xwb weights. CI64/CO64 is separately proven in [`ci64_co64_corrected_depthstride_proof.md`](ci64_co64_corrected_depthstride_proof.md). These fixtures do not prove CI128, other modes, or device behavior.

## Reproduction

From the repository root:

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding \
  -fno-builtin -fno-stack-protector -Wall -Wextra \
  native_kernels/qnn_hmx_generic_xwb_ci64_co32_correctdesc_sim.c \
  native_kernels/qd_qnn_hmx_stride1_generic.S \
  native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o native_kernels/qnn_hmx_generic_xwb_ci64_co32_correctdesc_sim.elf

docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim \
  -mv75 --quiet native_kernels/qnn_hmx_generic_xwb_ci64_co32_correctdesc_sim.elf
```

The retained log is [`qnn_hmx_generic_xwb_ci64_co32_correctdesc_sim.log`](qnn_hmx_generic_xwb_ci64_co32_correctdesc_sim.log). It reports zero mismatches for all nine taps and ends with `PASS CI64/CO32: raw Xwb off-diagonal depth tile, all 9 taps, full logical outputs, guards, and table tails.`

## Provenance

- Fixture C: `f66bfaa5cabed990154d12c04155bfef819e3f09f4261b546357e7e183c82758`
- Simulator ELF: `6713798f1678ca459aa3a512ea5d65c153094644dd642b13b5cfaded25c8ca6b`
- Simulator log: `cb0f4e8b71b2016a9e1a3d43e33422eb4a6c33fcfe4d96be4ee4a3a900347d0b`
- Extracted generic leaf assembly: `964ec477c48ce528e8dfef7cbd51925f8ebedc15af51e5c915e18f95de32cb05`
- Extracted stride-1 setter assembly: `7b18d8a11bb0e7eb278ddc2d1590acf9356ae9498148beb4d6f621648a342bf6`
- Finish helper assembly source: `ae30962a6a11a93a51f33ccf11f74312d316f2e2e8f97a59865060092e5612f1`
