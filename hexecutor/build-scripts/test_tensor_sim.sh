#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE}"
mkdir -p hexecutor/native/build/tensor
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
$tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 \
  hexecutor/tests/test_tensor_hmx_sim.c hexecutor/native/tensor/hmx_gemm.S \
  -o hexecutor/native/build/tensor/test_hmx.elf
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/tensor/test_hmx.elf > hexecutor/native/build/tensor/test_hmx.log
cat hexecutor/native/build/tensor/test_hmx.log
grep -q "^PASS fp16 HMX deep-load" hexecutor/native/build/tensor/test_hmx.log
'
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
$tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  hexecutor/tests/test_tensor_hmx_kernels_sim.c hexecutor/native/tensor/hmx_gemm.S \
  -o hexecutor/native/build/tensor/test_kernels.elf
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/tensor/test_kernels.elf > hexecutor/native/build/tensor/test_kernels.log
cat hexecutor/native/build/tensor/test_kernels.log
grep -q "^PASS fp16 conversion" hexecutor/native/build/tensor/test_kernels.log
'
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
$tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -ffp-contract=off \
  hexecutor/tests/test_tensor_fp16_sim.c hexecutor/native/tensor/graph.c \
  hexecutor/native/tensor/hmx.c hexecutor/native/tensor/fp16.c hexecutor/native/tensor/fp16_depthwise.S hexecutor/native/tensor/hmx_gemm.S \
  -o hexecutor/native/build/tensor/test_fp16.elf
$tools/bin/hexagon-sim -mv75 hexecutor/native/build/tensor/test_fp16.elf > hexecutor/native/build/tensor/test_fp16.log
cat hexecutor/native/build/tensor/test_fp16.log
grep -q "^PASS blocked FP16" hexecutor/native/build/tensor/test_fp16.log
'
