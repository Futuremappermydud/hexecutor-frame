#!/bin/sh
set -eu
cd "$(dirname "$0")/../../../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra \
  hexecutor/native_kernels/qnn_hmx_generic_xwb_qdrx247_ci64_sim.c \
  hexecutor/native_kernels/qd_qnn_hmx_stride1_generic.S \
  hexecutor/native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o hexecutor/native_kernels/qnn_hmx_generic_xwb_qdrx247_ci64_sim.elf
