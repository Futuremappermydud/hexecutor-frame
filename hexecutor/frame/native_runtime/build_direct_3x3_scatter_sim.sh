#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
FIXTURE=${1:-/src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin}
docker run --rm --platform linux/amd64 \
  -v "$PWD/../.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -DQD_DIRECT_ALLOW_MOCK_BASE=1 \
  -DQD_DIRECT_VTCM_BASE=0xd9000000u \
  -DQD_DIRECT_VTCM_BYTES=0x80000u -DQD_DIRECT_ACTIVE_BYTES=0x60000u \
  -DQD_DIRECT_TAIL_OFFSET=0x60000u -DQD_DIRECT_TAIL_BYTES=0x20000u \
  "-DQD_FIXTURE_PATH=\"$FIXTURE\"" \
  native_runtime/direct_3x3_scatter_sim.c native_runtime/qd_native_direct.c \
  native_kernels/qd_native_3x3_adapter.c native_kernels/qd_qnn_hmx_stride1_generic.S \
  native_kernels/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o native_runtime/direct_3x3_scatter_sim.elf
