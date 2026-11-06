#!/bin/sh
# Only the new probe; never build/replace the callback loader or default engine.
set -eu
cd "$(dirname "$0")/../.."
docker run --rm --pull never --platform linux/amd64 -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD:/workspace" -w /workspace/hexecutor/native "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -fno-stack-protector \
  -Wall -Wextra -ffunction-sections -fdata-sections -nostdlib -Wl,--gc-sections,-T,qd_payload.ld \
  ../benchmarks/qnn_hmx_1x1_frame.c qd_qnn_hmx_1x1.S qd_hmx_ops_encoded.S qd_graph_stack.S \
  -o build/qnn_hmx_1x1_frame.elf
