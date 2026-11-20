#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../native"
FIXTURE=${1:-/src/kernels/convolution/xwb_ci64_qdrx247_cpu_prefix_fixture.bin}
docker run --rm --platform linux/amd64 \
  -v "$PWD/../.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -DQD_DIRECT_ALLOW_MOCK_BASE=1 \
  -DQD_GRAPH_DIRECT_CALLER=1 \
  -DQD_DIRECT_VTCM_BASE=0xd9000000u \
  -DQD_DIRECT_VTCM_BYTES=0x80000u -DQD_DIRECT_ACTIVE_BYTES=0x60000u \
  -DQD_DIRECT_TAIL_OFFSET=0x60000u -DQD_DIRECT_TAIL_BYTES=0x20000u \
  "-DQD_FIXTURE_PATH=\"$FIXTURE\"" \
  ../research/provider-runtime/direct_3x3_scatter_sim.c ../research/provider-runtime/qd_native_direct.c \
  kernels/convolution/qd_native_3x3_adapter.c kernels/convolution/qd_qnn_hmx_stride1_generic.S \
  kernels/convolution/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o ../research/provider-runtime/direct_3x3_graph_direct_sim.elf
