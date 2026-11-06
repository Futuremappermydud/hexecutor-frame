#!/bin/sh
# Usage: bench_prepared_overhead_sim.sh [reps]
# Executor control-loop overhead on both real schedules in the V75 simulator
# (pcycles): stubbed handlers (trivial, tensor-decoding) and the real scalar
# reference validators. Needs frame/build/resident-{primary,alternate}.qdrx and
# recovered/<model>/constants. HEXAGON_SDK_HOST as in run_prepared_reference_sim.sh.
set -eu
cd "$(dirname "$0")/../.."
REPS=${1:-1}
SDK_MOUNT=""
if [ -n "${HEXAGON_SDK_HOST:-}" ]; then SDK_MOUNT="-v $HEXAGON_SDK_HOST:/repo/.work/hexagon-sdk:ro"; fi
docker run --rm --platform linux/amd64 $SDK_MOUNT -e REPS="$REPS" \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-overhead.elf
    "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffp-contract=off -fno-fast-math -fno-strict-aliasing \
      -Wall -Wextra -Werror -DBENCH_REAL -I"$SDK/target/hexagon/include" -Iframe/native_program \
      frame/native_program/bench_prepared_overhead.c \
      frame/native_program/prepared_reference.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/nprg2_runtime.c -lhexagon -lm -o "$ELF"
    exec "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF" -- \
      frame/build/resident-primary.qdrx frame/native_program/examples/primary-v2.nprg \
      frame/build/resident-alternate.qdrx frame/native_program/examples/alternate-v2.nprg "$REPS"
  '
