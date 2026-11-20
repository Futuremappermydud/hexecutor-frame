#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
  depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -DS2_BASIS_FIXTURE -DS2_UNIT_SCALES \
  hexecutor/research/provider-runtime/stride2/qnn_v69_s2_capture.c \
  hexecutor/research/provider-runtime/stride2/qd_v69_s2_dtype_stubs.S \
  hexecutor/research/provider-runtime/stride2/qd_v69_s2_helper_capture.S hexecutor/native/qd_hmx_ops.S \
  -o hexecutor/research/provider-runtime/stride2/qnn_v69_s2_basis.elf
for spec in 0,0,0 0,1,0 0,2,0 1,0,0 1,1,1 1,2,1 2,0,2 2,1,2 2,2,3; do
  IFS=, read -r ky kx cg <<EOF
$spec
EOF
  .venv-cpu/bin/python hexecutor/research/provider-runtime/stride2/make_basis_fixture.py "$ky" "$kx" --channel-group "$cg" --lane-grid
  docker run --rm --platform linux/amd64 \
    -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
    depth-model-hmx-tools:latest \
    /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
    hexecutor/research/provider-runtime/stride2/qnn_v69_s2_basis.elf \
    > "hexecutor/research/provider-runtime/stride2/lane_basis_${ky}_${kx}.log"
  cp hexecutor/research/provider-runtime/stride2/v69_s2_output_224.bin \
     "hexecutor/research/provider-runtime/stride2/lane_output_${ky}_${kx}.bin"
done
