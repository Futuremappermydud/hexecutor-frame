#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
if [ ! -x .venv-cpu/bin/python ]; then
  echo "Missing .venv-cpu/bin/python" >&2
  exit 1
fi
for model in primary alternate; do
  for pad in 0 3 7; do
    if [ ! -f "frame/native_program/fixtures/$model-smoke-reshape-cb-pad$pad.epcpu" ]; then
      .venv-cpu/bin/python frame/native_program/capture_reshape_cb_cpu_prefix.py --all-pads
      break 2
    fi
  done
done
docker run --rm --platform linux/amd64 \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-reshape-cb.elf
    "$SDK/bin/hexagon-clang" -mv75 -O2 -G0 -DRESHAPE_SIM_SUBSET \
      -Wall -Wextra -Werror -I"$SDK/target/hexagon/include" -Iframe/native_program \
      frame/native_program/test_prepared_reshape_cb.c \
      frame/native_program/prepared_reshape_cb.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/nprg2_runtime.c -lm -o "$ELF"
    "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
  '
