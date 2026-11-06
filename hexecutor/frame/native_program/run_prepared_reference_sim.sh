#!/bin/sh
# Usage: run_prepared_reference_sim.sh [primary|alternate|both] [prefix|full|segments] [segments per model] [fast] [cycles]
# prefix takes a few minutes. full runs each whole schedule in one simulator
# process and takes hours. segments splits each schedule on the host (default
# 6 segments per model, cut by host run time), then runs every segment in its
# own simulator process in parallel; together they cover the whole schedule.
# fast registers the fast exact convolution (HVX vrmpy in the simulator);
# ew registers the HVX elementwise handlers (prepared_elementwise.c) for scratch
# region 1 and chains the reference kernel behind them, as the Frame build does;
# cycles prints simulator pcycles per (opcode, variant).
# SEGDIR (default .work/full/segments) holds the segment files and logs; give a
# concurrent run its own SEGDIR, since a segments run deletes the old logs.
# SEGMENT_GLOB (default *) limits which segment files are simulated.
# HEXAGON_SDK_HOST, when set, is a host directory mounted read-only at
# /repo/.work/hexagon-sdk (for a worktree that has no SDK of its own).
set -eu
cd "$(dirname "$0")/../.."
WHICH=${1:-both}
MODE=${2:-prefix}
SEGMENTS=${3:-6}
if [ $# -gt 3 ]; then shift 3; EXTRA="$*"; else EXTRA=""; fi
N=frame/native_program
SEGDIR=${SEGDIR:-.work/full/segments}
SEGMENT_GLOB=${SEGMENT_GLOB:-*}
if [ "$MODE" = prefix ]; then
  for model in primary alternate; do
    case $model in primary) n=132 ;; alternate) n=100 ;; esac
    if [ ! -f "$N/fixtures/$model-smoke-prefix-$n.mpfx" ]; then
      if [ ! -x .venv-cpu/bin/python ]; then
        echo "Missing .venv-cpu/bin/python; create the project CPU environment to capture the NativeGraph CPU prefixes." >&2
        exit 1
      fi
      .venv-cpu/bin/python $N/capture_multi_prefix_cpu.py --model "$model"
    fi
  done
fi
if [ "$MODE" = segments ]; then
  mkdir -p "$SEGDIR"
  rm -f "$SEGDIR"/*.rseg "$SEGDIR"/*.log
  host="$SEGDIR/test_refsim_split"
  cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror ${CFLAGS_EXTRA:-} -I$N \
    $N/test_prepared_reference_sim.c $N/prepared_reference.c $N/prepared_executor.c $N/prepared_fuse.c $N/nprg2_runtime.c \
    -lm -o "$host"
  for model in primary alternate; do
    if [ "$WHICH" = both ] || [ "$WHICH" = "$model" ]; then
      "$host" "$model" split "$SEGMENTS" "$SEGDIR" $EXTRA | grep -v opcodes
    fi
  done
fi
# CFLAGS_EXTRA (optional): extra compiler flags, e.g. -DDM_ENABLED=0 -DQD_NO_VCOPY for the pre-fast-path baseline.
# SDK_MOUNT or HEXAGON_SDK_HOST (optional): host path of .work/hexagon-sdk, for a worktree without one.
SDK_MOUNT=${SDK_MOUNT:-${HEXAGON_SDK_HOST:-}}
SDKARG=""
if [ -n "$SDK_MOUNT" ]; then SDKARG="-v $SDK_MOUNT:/repo/.work/hexagon-sdk:ro"; fi
docker run --rm --platform linux/amd64 $SDKARG ${DOCKER_EXTRA:-} \
  -e CFX="${CFLAGS_EXTRA:-}" -e WHICH="$WHICH" -e MODE="$MODE" -e SEGDIR="$SEGDIR" -e SEGMENT_GLOB="$SEGMENT_GLOB" -e EXTRA="$EXTRA" \
  -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc '
    SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    ELF=/tmp/nprg2-reference.elf
    "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffp-contract=off -fno-fast-math -fno-strict-aliasing \
      -Wall -Wextra -Werror $CFX -I"$SDK/target/hexagon/include" -Iframe/native_program \
      frame/native_program/test_prepared_reference_sim.c \
      frame/native_program/prepared_reference.c \
      frame/native_program/prepared_executor.c \
      frame/native_program/prepared_fuse.c \
      frame/native_program/prepared_elementwise.c \
      frame/blocked_glue/add_u8.c \
      frame/blocked_glue/linearclip_u8_u16.c \
      frame/blocked_glue/elementwise_u16.c \
      frame/blocked_glue/linearclip_u16.c \
      frame/blocked_glue/sign_levels.c \
      frame/blocked_glue/sign_blocked.c \
      frame/native_program/nprg2_runtime.c frame/qd_hmx_ops_encoded.S -lhexagon -lm -o "$ELF"
    if [ "$MODE" != segments ]; then
      exec "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF" -- "$WHICH" "$MODE" $EXTRA
    fi
    pids=""
    for f in "$SEGDIR"/$SEGMENT_GLOB.rseg; do
      "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF" -- segment "$f" 50 $EXTRA > "${f%.rseg}.log" 2>&1 &
      pids="$pids $!"
    done
    status=0
    for p in $pids; do wait "$p" || status=1; done
    for f in "$SEGDIR"/$SEGMENT_GLOB.rseg; do
      log="${f%.rseg}.log"
      if grep -q "^PASS" "$log"; then grep "^PASS" "$log"; else echo "FAILED: $log"; cat "$log"; status=1; fi
    done
    exit $status
  '
