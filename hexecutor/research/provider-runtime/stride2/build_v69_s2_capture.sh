#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../../.."
capture_flag=
if [ "${S2_CAPTURE_ONLY:-0}" = 1 ]; then capture_flag=-DS2_CAPTURE_ONLY; fi
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" \
  -v "$PWD/hexecutor/native/kernels/stride2:/src/native_stride2:ro" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra \
  $capture_flag \
  hexecutor/research/provider-runtime/stride2/qnn_v69_s2_capture.c \
  hexecutor/research/provider-runtime/stride2/qd_v69_s2_dtype_stubs.S \
  hexecutor/research/provider-runtime/stride2/qd_v69_s2_helper_capture.S \
  hexecutor/native/qd_hmx_ops.S \
  -o hexecutor/research/provider-runtime/stride2/qnn_v69_s2_capture.elf
