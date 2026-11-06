#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
test -f frame/native_adapter/fixture-qdrx-1195.bin
./frame/native_runtime/run_direct_binding_test.sh
./frame/native_runtime/build_direct_sim.sh /src/native_adapter/fixture-qdrx-1195.bin
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
  native_runtime/direct_1x1_sim.elf
python3 frame/native_runtime/check_direct_1x1_sim.py
