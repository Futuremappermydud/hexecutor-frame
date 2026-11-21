#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Builds the Q16P engine (hexecutor/native/qnn16) for the shared tensor
# bootstrap. The SDK path and installed image are explicit. Run
# build_tensor.sh first: the session uses its bootstrap and chunk probe.
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT to the Hexagon SDK root}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE to an installed toolchain image}"
slots="${HEXECUTOR_QDO_SLOTS:-8}"
out=qnn16
[ "$slots" = 8 ] || out="qnn16-$slots"
mkdir -p "hexecutor/native/build/$out"
docker run --rm --pull never --platform linux/amd64 -e QNN16_DEFINES="${QNN16_DEFINES:-}" -e SLOTS="$slots" -e OUT="$out" \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" \
  -w /workspace/hexecutor/native "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
cc=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
flags="-mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -ffp-contract=off -fno-strict-aliasing -Wall -Wextra -DQDO_SCALAR_CACHE=0 -DQDO_SLOTS=$SLOTS $QNN16_DEFINES"
# No autovectorization: scratch is L1-cached and HVX accesses bypass L1.
$cc $flags -fno-vectorize -fno-slp-vectorize -ffunction-sections -fdata-sections -nostdlib -Wl,--gc-sections,-T,qd_graph_dsp.ld \
  qnn16/engine.c qnn16/program.c qnn16/exec.c qnn16/qnn16_f32.c qnn16/qnn16_hvx.c qnn16/qnn16_tile*.c qnn16/qnn16_hmx.S tensor/session.c \
  -Wl,--start-group -lhexagon -lgcc -lc -lm -Wl,--end-group -o build/$OUT/engine.elf
'
