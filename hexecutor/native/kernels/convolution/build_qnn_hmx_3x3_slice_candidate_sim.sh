#!/bin/sh
set -eu
cd "$(dirname "$0")/../../../.."
FIXTURE=${1:-/src/hexecutor/native_kernels/xwb_ci128_qdrx3627_slice_cpu_prefix_fixture.bin}
task_slice_core=${QD_SLICE_CORE:-hexecutor/native_kernels/qd_native_3x3_slice_candidate.c}
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD/frame:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra "-DQD_FIXTURE_PATH=\"$FIXTURE\"" \
  "-DQDS_NATIVE_3X3_SLICES=${QDS_NATIVE_3X3_SLICES:-0}" \
  hexecutor/native_kernels/qnn_hmx_3x3_slice_candidate_sim.c \
  "$task_slice_core" \
  hexecutor/native_kernels/qd_qnn_hmx_stride1_generic.S hexecutor/native_kernels/qd_set_hmx_params_stride1.S \
  qd_hmx_ops.S -o hexecutor/native_kernels/qnn_hmx_3x3_slice_candidate_sim.elf
