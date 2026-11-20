#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# The SDK path and installed image are explicit; no model-resource aliases.
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT to the Hexagon SDK root}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE to an installed toolchain image}"
lqdma="${HEXECUTOR_LQ_DMA:-1}" # LINEAR_Q weights streamed by user DMA into VTCM
slots="${HEXECUTOR_QDO_SLOTS:-8}" # 16 MiB JTLB entries mapped by the engine: 8 = 128 MiB, 64 = 1024 MiB
out=tensor; [ "$slots" = 8 ] || out=tensor-$slots
mkdir -p hexecutor/native/build/$out
docker run --rm -e SLOTS="$slots" -e LQDMA="$lqdma" -e EXTRA="${HEXECUTOR_ENGINE_CFLAGS:-}" -e OUT="$out" --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" \
  -w /workspace/hexecutor/native "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
cc=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
flags="-mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -ffp-contract=off -fno-strict-aliasing -Wall -Wextra -DQDO_SCALAR_CACHE=0 -DQDO_SLOTS=$SLOTS"
$cc $flags -DHX_LINEAR_Q_DMA=$LQDMA ${EXTRA:-} -ffunction-sections -fdata-sections -nostdlib -Wl,--gc-sections,-T,qd_graph_dsp.ld \
  tensor/engine.c tensor/graph.c tensor/session.c tensor/hmx.c tensor/fp16.c tensor/fp16_depthwise.S tensor/hmx_gemm.S -Wl,--start-group -lhexagon -lgcc -lc -lm -Wl,--end-group -o build/$OUT/engine.elf
$cc $flags -fno-vectorize -fno-slp-vectorize -DHEXECUTOR_TENSOR_BOOT=1 \
  -DQDS_PRESERVE_VTCM=1 -DQDS_ENTRY_GUARD=1 -DQDS_AXIM_RUN=1 -DQDS_PREPARED_THREADS=4 \
  -nostdlib -Wl,-T,qd_payload.ld \
  qd_graph_boot.c qd_graph_stack.S qd_hvx_context.S tensor/session.c \
  -Wl,--start-group -lhexagon -lgcc -lc -lm -Wl,--end-group -o build/$OUT/bootstrap.elf
$cc $flags -nostdlib -Wl,-T,qd_payload.ld ../benchmarks/qd_chunk_map_probe.c -o build/$OUT/chunk_probe.elf
'
