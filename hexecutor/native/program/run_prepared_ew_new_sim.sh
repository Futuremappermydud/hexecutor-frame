#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Focused V75 simulator tests for the SIGN/39, LINEARCLIP/67 and LINEARCLIP/69 elementwise handlers.
#   1. hexecutor/blocked_glue/linearclip_flat_u8_sim.c: the flat u8 kernel against the independent int64 expression.
#   2. Windows of the real schedules around those commands: the host reference runs the schedule up to the
#      window and through it; the simulator starts from the window's start state with the HVX handlers
#      (ew) chained over the reference and must reproduce the end state bit for bit. With "cycles" it prints
#      pcycles per (opcode, variant). Windows (command ranges) are listed below.
# The windows also cover SPILL_FILL (26/8, 26/9), CONCAT/15, INPUT_SLICE/1 and OUTPUT_SLICE/3,4.
# SDK_MOUNT: host path of .work/hexagon-sdk (for a worktree without its own). SEGDIR: scratch dir.
set -eu
cd "$(dirname "$0")/../../.."
N=hexecutor/native/program
SEGDIR=${SEGDIR:-.work/ewnew}
SDK_MOUNT=${SDK_MOUNT:-${HEXAGON_SDK_HOST:-}}
SDKARG=""
if [ -n "$SDK_MOUNT" ]; then SDKARG="-v $SDK_MOUNT:/repo/depth-model/.work/hexagon-sdk:ro"; fi
mkdir -p "$SEGDIR"
host="$SEGDIR/test_refsim_split"
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror -I$N \
  $N/test_prepared_reference_sim.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c -lm -o "$host"
rm -f "$SEGDIR"/*.rseg "$SEGDIR"/*.log
# model first count
for w in "primary 4584 4" "primary 4776 4" "primary 4805 1" "primary 4808 11" "primary 4875 11" "primary 4932 22" "primary 2982 9" \
         "primary 968 12" "primary 2039 8" "primary 2924 1" "primary 2955 1" "primary 2991 1" "primary 4468 1" "primary 4391 1" "primary 5060 5" \
         "alternate 4666 3" "alternate 4681 12" "alternate 4758 13" "alternate 4833 6" "alternate 4885 32" "alternate 2938 9" \
         "alternate 1015 4" "alternate 1741 4" "alternate 2893 1" "alternate 2918 1" "alternate 2950 1" "alternate 4421 1" "alternate 4341 1" "alternate 5032 5"; do
  set -- $w
  mkdir -p "$SEGDIR/$1-$2"
  "$host" "$1" tail "$2" "$SEGDIR/$1-$2" "$3" | grep -v "^PASS"
done
docker run --rm --platform linux/amd64 $SDKARG ${DOCKER_EXTRA:-} -e SEGDIR="$SEGDIR" -e CFX="${CFLAGS_EXTRA:-}" -e SIMARGS="${SIMARGS:-fast ew cycles}" \
  -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
    SDK=/repo/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
    CC="$SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -Wall -Wextra -Werror $CFX -I$SDK/target/hexagon/include"
    $CC hexecutor/native/kernels/blocked/linearclip_flat_u8_sim.c -o /tmp/lcflat.elf -lhexagon
    "$SDK/bin/hexagon-sim" -mv75 --quiet /tmp/lcflat.elf
    $CC -Ihexecutor/native/program hexecutor/native/program/test_prepared_reference_sim.c hexecutor/native/program/prepared_reference.c \
      hexecutor/native/program/prepared_executor.c hexecutor/native/program/prepared_elementwise.c hexecutor/native/kernels/blocked/add_u8.c \
      hexecutor/native/kernels/blocked/linearclip_u8_u16.c hexecutor/native/kernels/blocked/elementwise_u16.c hexecutor/native/kernels/blocked/linearclip_u16.c \
      hexecutor/native/kernels/blocked/sign_levels.c hexecutor/native/kernels/blocked/sign_blocked.c hexecutor/native/program/nprg2_runtime.c -lhexagon -lm -o /tmp/ewnew.elf
    status=0
    for f in "$SEGDIR"/*/*.rseg; do
      "$SDK/bin/hexagon-sim" -mv75 --quiet /tmp/ewnew.elf -- segment "$f" 50 $SIMARGS > "${f%.rseg}.log" 2>&1 &
    done
    wait
    for f in "$SEGDIR"/*/*.rseg; do
      log="${f%.rseg}.log"
      if grep -q "^PASS" "$log"; then grep "^PASS" "$log"; else echo "FAILED: $log"; cat "$log"; status=1; fi
    done
    exit $status
  '
