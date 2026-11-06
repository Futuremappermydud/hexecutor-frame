#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
python3 frame/native_runtime/make_direct_3x3_scatter_map.py
./frame/native_runtime/build_direct_3x3_scatter_sim.sh /src/native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
  native_runtime/direct_3x3_scatter_sim.elf
python3 frame/native_runtime/check_direct_3x3_scatter_sim.py
