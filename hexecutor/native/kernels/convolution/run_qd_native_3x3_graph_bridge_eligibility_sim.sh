#!/bin/sh
set -eu
cd "$(dirname "$0")/../../../.."
.venv-cpu/bin/python hexecutor/native/kernels/convolution/generate_native3_bridge_cases.py
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -I. -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra \
  hexecutor/native_kernels/qd_native_3x3_graph_bridge_eligibility_sim.c \
  -o hexecutor/native_kernels/qd_native_3x3_graph_bridge_eligibility_sim.elf
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
  hexecutor/native_kernels/qd_native_3x3_graph_bridge_eligibility_sim.elf
