#!/bin/sh
# Builds only the isolated probe ELF. Never rebuilds or replaces qd_loader.elf.
set -eu
cd "$(dirname "$0")/../../../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
  -nostdlib -Wl,--gc-sections,-T,qd_payload.ld \
  hexecutor/native_kernels/xwb1022_frame_probe.c hexecutor/native_kernels/qd_qnn_hmx_stride1_generic.S \
  hexecutor/native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops_encoded.S qd_graph_stack.S \
  -o hexecutor/native_kernels/xwb1022_frame_probe.elf
