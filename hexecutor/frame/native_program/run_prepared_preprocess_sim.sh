#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
.venv-cpu/bin/python frame/native_program/capture_preprocess_cpu_prefix.py --case smoke --out frame/native_program/fixtures
docker run --rm --platform linux/amd64 \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-preprocess.elf
    "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 \
      -Wall -Wextra -Werror -I"$SDK/target/hexagon/include" \
      frame/native_program/test_prepared_preprocess.c \
      frame/native_program/prepared_preprocess.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/nprg2_runtime.c -o "$ELF"
    "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
  '
