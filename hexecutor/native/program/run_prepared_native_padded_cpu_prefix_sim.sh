#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD:/src" -w /src "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
    SDK=/src/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    for CASE in 1 2 3 4; do
      ELF=/tmp/prepared-native-padded-$CASE.elf
      "$SDK/bin/hexagon-clang" -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 \
        -Wall -Wextra -Werror -DQD_DIRECT_ALLOW_MOCK_BASE=1 -DQDS_NATIVE_PADDED_1X1=1 -DNCPF_CASE=$CASE \
        -I"$SDK/target/hexagon/include" \
        hexecutor/native/program/test_prepared_native_padded_cpu_prefix_sim.c \
        hexecutor/native/program/prepared_native.c \
        hexecutor/native/program/prepared_executor.c \
        hexecutor/native/program/nprg2_runtime.c \
        hexecutor/research/provider-runtime/qd_native_direct.c \
        hexecutor/native/qd_qnn_hmx_1x1.S hexecutor/native/qd_hmx_ops.S \
        hexecutor/native/kernels/convolution/qd_set_hmx_params_stride1.S -o "$ELF" && \
      "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
    done
  '
