# CI64/CO64 raw-Xwb generic leaf proof

This extends the CI64/CO32 simulator proof to two input depth blocks and two output channel blocks. It also records the FI layout issue found while isolating the first CO64 failure.

## Proven geometry and FI layout

- Input: logical 10×10×64, seven-pixel top/left origin, 3×3 spatial crouton grid, and 2 depth blocks per crouton. Descriptor: `{table, 2, 6, 17, 17, 64}`.
- Output: logical 8×8×64, one spatial crouton and 2 output depth blocks. Descriptor: `{table, 2, 2, 8, 8, 64}`.
- Raw Xwb bytes follow `[O/32,I/32,H,W,(I%32)/4,O%32,I%4]`.
- FI layout is per output depth block: `[multiplier[32], bias[32]]` repeated. A global `[multiplier[64], bias[64]]` arrangement made the first output depth block interpret scale words as bias and saturated every channel; initializing each 64-word block separately removed that failure.
- Fixture makes the cross-group permutation `CO=(CI+17)%64` with one weight at a selected stored filter tap. This exercises both input depth blocks, both output blocks, and off-diagonal channel terms.
- Each of all nine taps matches the independent physical input oracle for all `8×8×64 = 4,096` output bytes. Input/output guards, table tails, and finish drain pass. Completion polling spins are 0.
- Finish zero-FI is 64 words, with 32 scale words followed by 32 zero biases. This is the 32-channel finish helper, separate from the 128-word CO64 convolution FI.

The simulator proves this bounded generic CI64/CO64 shape. It does not establish CI128, alternate output modes, or device execution.

## Reproduction

From the repository root:

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding \
  -fno-builtin -fno-stack-protector -Wall -Wextra \
  native_kernels/qnn_hmx_generic_xwb_ci64_co64_correctdesc_sim.c \
  native_kernels/qd_qnn_hmx_stride1_generic.S \
  native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o native_kernels/qnn_hmx_generic_xwb_ci64_co64_correctdesc_sim.elf

docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim \
  -mv75 --quiet native_kernels/qnn_hmx_generic_xwb_ci64_co64_correctdesc_sim.elf
```

Retained output is [`qnn_hmx_generic_xwb_ci64_co64_correctdesc_sim.log`](qnn_hmx_generic_xwb_ci64_co64_correctdesc_sim.log): each tap reports `mismatches=0/4096`, and the final line is `PASS CI64/CO64 cross-depth Xwb permutation, all 9 taps, full outputs, guards, and table tails.` The isolated one-hot FI diagnosis is retained in [`qnn_hmx_generic_xwb_ci64_co64_onehot_sim.log`](qnn_hmx_generic_xwb_ci64_co64_onehot_sim.log).

## Provenance

- All-tap fixture C: `8f5dae13fd21e0fc1d67017624aa6460686563ece54746cff59ae34d72dc9dfd`
- All-tap simulator ELF: `5e8f4b855415346311440a97dc21f573c56f8c2c81d74409bec707b9eebcc22e`
- All-tap simulator log: `64b614003cbdc633e41f7660226de4597530b8bea25fd7e416f889ea95d7de78`
- One-hot FI diagnosis C: `00dceca82b02816278c054cfe7329ad4653ea464fe202196f692931f6a2e5403`
- One-hot simulator ELF: `b14f0f45902f353847168941c3c192b7191ebb5040d2917317b4ebf4393d37ea`
- One-hot simulator log: `981ce335236b141333455b5ad39fac225148c5c441551873f174e09d93c342c2`
- Extracted generic leaf assembly: `964ec477c48ce528e8dfef7cbd51925f8ebedc15af51e5c915e18f95de32cb05`
- Extracted stride-1 setter assembly: `7b18d8a11bb0e7eb278ddc2d1590acf9356ae9498148beb4d6f621648a342bf6`
- Finish helper assembly source: `ae30962a6a11a93a51f33ccf11f74312d316f2e2e8f97a59865060092e5612f1`
