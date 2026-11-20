#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../native"
FIXTURE=${1:-/src/kernels/adapters/fixture-qdrx-1195.bin}
docker run --rm --platform linux/amd64 \
  -v "$PWD/../.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -DQD_DIRECT_ALLOW_MOCK_BASE=1 \
  -DQD_DIRECT_VTCM_BYTES=0x80000u -DQD_DIRECT_ACTIVE_BYTES=0x60000u \
  -DQD_DIRECT_TAIL_OFFSET=0x60000u -DQD_DIRECT_TAIL_BYTES=0x20000u \
  "-DQD_FIXTURE_PATH=\"$FIXTURE\"" \
  ../research/provider-runtime/direct_1x1_sim.c ../research/provider-runtime/qd_native_direct.c kernels/adapters/qd_native_1x1_adapter.c \
  qd_qnn_hmx_1x1.S kernels/convolution/qd_set_hmx_params_stride1.S qd_hmx_ops.S \
  -o ../research/provider-runtime/direct_1x1_sim.elf
