#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
if [ ! -x .venv-cpu/bin/python ]; then
  echo "Missing .venv-cpu/bin/python; create the project CPU environment to capture the NativeGraph CPU prefixes." >&2
  exit 1
fi
.venv-cpu/bin/python frame/native_program/capture_abs_cpu_prefix.py \
  --case smoke --out frame/native_program/fixtures
docker run --rm --platform linux/amd64 \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-abs.elf
    "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 \
      -Wall -Wextra -Werror -I"$SDK/target/hexagon/include" -Iframe/native_program \
      frame/native_program/test_prepared_abs.c \
      frame/native_program/prepared_abs.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/nprg2_runtime.c -o "$ELF"
    "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
  '
