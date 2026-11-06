#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD:/src" -w /src depth-model-hmx-tools:latest sh -lc '
    SDK=/src/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    for CASE in 1 2 3 4; do
      ELF=/tmp/prepared-native-padded-$CASE.elf
      "$SDK/bin/hexagon-clang" -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 \
        -Wall -Wextra -Werror -DQD_DIRECT_ALLOW_MOCK_BASE=1 -DQDS_NATIVE_PADDED_1X1=1 -DNCPF_CASE=$CASE \
        -I"$SDK/target/hexagon/include" \
        frame/native_program/test_prepared_native_padded_cpu_prefix_sim.c \
        frame/native_program/prepared_native.c \
        frame/native_program/prepared_executor.c \
        frame/native_program/nprg2_runtime.c \
        frame/native_runtime/qd_native_direct.c \
        frame/qd_qnn_hmx_1x1.S frame/qd_hmx_ops.S \
        frame/native_kernels/qd_set_hmx_params_stride1.S -o "$ELF" && \
      "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
    done
  '
