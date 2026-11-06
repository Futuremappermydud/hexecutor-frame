#!/bin/sh
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
for ky in 1 2; do
  for kx in 0 1 2; do
    for cg in 0 1 2 3; do
      .venv-cpu/bin/python hexecutor/research/provider-runtime/stride2/make_basis_fixture.py "$ky" "$kx" --dense-input --channel-group "$cg"
      docker run --rm --platform linux/amd64 \
        -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
        depth-model-hmx-tools:latest \
        /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
        hexecutor/research/provider-runtime/stride2/qnn_v69_s2_basis.elf \
        > "hexecutor/research/provider-runtime/stride2/dense_channel_${ky}_${kx}_${cg}.log"
      cp hexecutor/research/provider-runtime/stride2/v69_s2_output_224.bin \
         "hexecutor/research/provider-runtime/stride2/dense_channel_output_${ky}_${kx}_${cg}.bin"
    done
  done
done
