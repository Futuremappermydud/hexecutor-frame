#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE}"

mkdir -p hexecutor/native/build/lfm
python3 hexecutor/examples/llm/lfm/gen_persistent_numpy_golden.py
python3 hexecutor/examples/llm/lfm/gen_hxfg_decode_golden.py hexecutor/native/build/lfm/hxfg_decode
# The same decode graph through the ONNX lowering; needs the offline onnx package (HEXECUTOR_PYTHON).
PY="${HEXECUTOR_PYTHON:-python3}"
if "$PY" -c "import onnx" 2>/dev/null; then
  "$PY" hexecutor/examples/llm/lfm/gen_onnx_decode_golden.py hexecutor/native/build/lfm/onnx_decode
  ONNX_LOWERING=1
else
  echo "SKIP ONNX lowering decode test: no onnx package in $PY (set HEXECUTOR_PYTHON)"
  ONNX_LOWERING=0
fi
# And through ExecuTorch (pinned venv: hexecutor/offline/executorch/README.md).
ET_PY="${HEXECUTOR_ET_PYTHON:-hexecutor/.work/executorch-venv/bin/python}"
if [ -x "$ET_PY" ] && "$ET_PY" -c "import executorch.exir" 2>/dev/null; then
  "$ET_PY" hexecutor/examples/llm/lfm/gen_executorch_decode_golden.py hexecutor/native/build/lfm/et_decode
  ET_LOWERING=1
else
  echo "SKIP ExecuTorch lowering decode test: no executorch venv at $ET_PY (set HEXECUTOR_ET_PYTHON)"
  ET_LOWERING=0
fi
docker run --rm --pull never --platform linux/amd64 -e ONNX_LOWERING="$ONNX_LOWERING" -e ET_LOWERING="$ET_LOWERING" \
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

$tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off -Wno-format \
  -Ihexecutor/native/tensor hexecutor/tests/test_linear_q_hvx_sim.c \
  -o hexecutor/native/build/lfm/test_linear_q_hvx_sim.elf -lm
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_linear_q_hvx_sim.elf > hexecutor/native/build/lfm/test_linear_q_hvx_sim.log
cat hexecutor/native/build/lfm/test_linear_q_hvx_sim.log
grep -q "^PASS linear_q hvx" hexecutor/native/build/lfm/test_linear_q_hvx_sim.log

$tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  -Ihexecutor/native/tensor \
  hexecutor/examples/llm/lfm/test_persistent_state_sim.c \
  hexecutor/native/tensor/lfm_ops.c \
  hexecutor/native/tensor/graph.c \
  -o hexecutor/native/build/lfm/test_persistent_sim.elf -lm
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_persistent_sim.elf > hexecutor/native/build/lfm/test_persistent_sim.log
cat hexecutor/native/build/lfm/test_persistent_sim.log
grep -q "^PASS persistent state sim" hexecutor/native/build/lfm/test_persistent_sim.log

$tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  -Ihexecutor/native/tensor -Ihexecutor/native/build/lfm/hxfg_decode \
  hexecutor/examples/llm/lfm/test_hxfg_decode_sim.c \
  hexecutor/native/tensor/graph.c \
  -o hexecutor/native/build/lfm/test_hxfg_decode_sim.elf -lm
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_hxfg_decode_sim.elf > hexecutor/native/build/lfm/test_hxfg_decode_sim.log
cat hexecutor/native/build/lfm/test_hxfg_decode_sim.log
grep -q "^PASS HXFG writer decode" hexecutor/native/build/lfm/test_hxfg_decode_sim.log

if [ "$ONNX_LOWERING" = 1 ]; then
  $tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
    -Ihexecutor/native/tensor -Ihexecutor/native/build/lfm/onnx_decode \
    hexecutor/examples/llm/lfm/test_hxfg_decode_sim.c \
    hexecutor/native/tensor/graph.c \
    -o hexecutor/native/build/lfm/test_onnx_decode_sim.elf -lm
  $tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_onnx_decode_sim.elf > hexecutor/native/build/lfm/test_onnx_decode_sim.log
  cat hexecutor/native/build/lfm/test_onnx_decode_sim.log
  grep -q "^PASS HXFG writer decode" hexecutor/native/build/lfm/test_onnx_decode_sim.log
fi

if [ "$ET_LOWERING" = 1 ]; then
  $tools/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
    -Ihexecutor/native/tensor -Ihexecutor/native/build/lfm/et_decode \
    hexecutor/examples/llm/lfm/test_hxfg_decode_sim.c \
    hexecutor/native/tensor/graph.c \
    -o hexecutor/native/build/lfm/test_et_decode_sim.elf -lm
  $tools/bin/hexagon-sim -mv75 hexecutor/native/build/lfm/test_et_decode_sim.elf > hexecutor/native/build/lfm/test_et_decode_sim.log
  cat hexecutor/native/build/lfm/test_et_decode_sim.log
  grep -q "^PASS HXFG writer decode" hexecutor/native/build/lfm/test_et_decode_sim.log
fi
'
