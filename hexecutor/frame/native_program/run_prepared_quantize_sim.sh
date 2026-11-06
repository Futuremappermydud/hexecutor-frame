#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
docker run --rm --platform linux/amd64 \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-quantize.elf
    "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 \
      -Wall -Wextra -Werror -DQDS_PREPARED_QUANTIZE=1 -I"$SDK/target/hexagon/include" \
      frame/native_program/test_prepared_quantize.c \
      frame/native_program/prepared_quantize.c \
      frame/native_program/prepared_quantize_core.c \
      frame/native_program/prepared_preprocess.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/nprg2_runtime.c -o "$ELF"
    "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
  '
