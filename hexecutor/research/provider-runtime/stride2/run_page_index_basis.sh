#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../../.."
for spec in 0,1,0,4,12 1,1,1,4,12 2,2,3,4,12 0,1,0,12,4 1,1,1,12,4 2,2,3,12,4 0,1,0,4,7 0,1,0,4,8; do
  IFS=, read -r ky kx cg ty tx <<EOF
$spec
EOF
  .venv-cpu/bin/python hexecutor/research/provider-runtime/stride2/make_basis_fixture.py "$ky" "$kx" --channel-group "$cg" --target-y "$ty" --target-x "$tx"
  docker run --rm --platform linux/amd64 \
    -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" -v "$PWD:/src" -w /src \
    depth-model-hmx-tools:latest \
    /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim -mv75 --quiet \
    hexecutor/research/provider-runtime/stride2/qnn_v69_s2_basis.elf \
    > "hexecutor/research/provider-runtime/stride2/page_basis_${ky}_${kx}_${ty}_${tx}.log"
  cp hexecutor/research/provider-runtime/stride2/v69_s2_output_224.bin \
     "hexecutor/research/provider-runtime/stride2/page_output_${ky}_${kx}_${ty}_${tx}.bin"
done
