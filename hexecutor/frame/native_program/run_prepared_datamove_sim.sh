#!/bin/sh
# Usage: run_prepared_datamove_sim.sh [cycles]
# Builds test_prepared_datamove_sim.c for V75 (HVX 128B) and runs it in
# hexagon-sim. SDK_MOUNT (optional): host path of .work/hexagon-sdk.
set -eu
cd "$(dirname "$0")/../.."
SDKARG=""
if [ -n "${SDK_MOUNT:-}" ]; then SDKARG="-v $SDK_MOUNT:/repo/.work/hexagon-sdk:ro"; fi
docker run --rm --platform linux/amd64 $SDKARG -e ARGS="$*" -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
  SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
  ELF=/tmp/nprg2-datamove.elf
  "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffp-contract=off -fno-fast-math -fno-strict-aliasing \
    -Wall -Wextra -Werror -I"$SDK/target/hexagon/include" -Iframe/native_program \
    frame/native_program/test_prepared_datamove_sim.c frame/native_program/prepared_executor.c \
    frame/native_program/nprg2_runtime.c -lhexagon -lm -o "$ELF"
  exec "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF" -- $ARGS'
