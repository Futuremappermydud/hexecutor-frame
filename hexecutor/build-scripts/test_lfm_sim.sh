#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE}"

mkdir -p hexecutor/native/build/lfm
python3 hexecutor/examples/llm/lfm/gen_persistent_numpy_golden.py
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
$tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  -Ihexecutor/native/tensor \
  hexecutor/examples/llm/lfm/test_lfm_decode_sim.c \
  hexecutor/native/tensor/lfm_ops.c \
  hexecutor/native/tensor/graph.c \
  -o hexecutor/native/build/lfm/test_lfm_sim.elf -lm
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_lfm_sim.elf > hexecutor/native/build/lfm/test_lfm_sim.log
cat hexecutor/native/build/lfm/test_lfm_sim.log
grep -q "^PASS LFM synthetic decode" hexecutor/native/build/lfm/test_lfm_sim.log
grep -q "^PASS Qwen regression check" hexecutor/native/build/lfm/test_lfm_sim.log

$tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  -Ihexecutor/native/tensor \
  hexecutor/examples/llm/lfm/test_persistent_state_sim.c \
  hexecutor/native/tensor/lfm_ops.c \
  hexecutor/native/tensor/graph.c \
  -o hexecutor/native/build/lfm/test_persistent_sim.elf -lm
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_persistent_sim.elf > hexecutor/native/build/lfm/test_persistent_sim.log
cat hexecutor/native/build/lfm/test_persistent_sim.log
grep -q "^PASS persistent state sim" hexecutor/native/build/lfm/test_persistent_sim.log
'
