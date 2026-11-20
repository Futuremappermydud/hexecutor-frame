#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../.."
python3 hexecutor/research/provider-runtime/make_direct_3x3_scatter_map.py
hexecutor/research/provider-runtime/build_direct_3x3_scatter_sim.sh /src/hexecutor/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
  hexecutor/native_runtime/direct_3x3_scatter_sim.elf
python3 hexecutor/research/provider-runtime/check_direct_3x3_scatter_sim.py
