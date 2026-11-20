#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../.."
test -f hexecutor/native/kernels/adapters/fixture-qdrx-1195.bin
hexecutor/research/provider-runtime/run_direct_binding_test.sh
hexecutor/research/provider-runtime/build_direct_sim.sh /src/hexecutor/native_adapter/fixture-qdrx-1195.bin
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
  hexecutor/native_runtime/direct_1x1_sim.elf
python3 hexecutor/research/provider-runtime/check_direct_1x1_sim.py
