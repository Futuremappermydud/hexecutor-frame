#!/bin/sh
# Separate experimental payload. Never rebuild the loader or replace v5.
set -eu
cd "$(dirname "$0")/.."
task_sdk_root="$PWD/.work/hexagon-sdk/6.6.0.0"
docker run --rm --platform linux/amd64 -v "$task_sdk_root:/sdk:ro" -v "$PWD/frame:/src" -w /src \
 depth-model-hmx-tools:latest sh -eu -c '
cc=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
flags="-mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -ffp-contract=off -fno-strict-aliasing -Wall -Wextra"
$cc $flags -DQDC_NATIVE_BUILD=1 -ffunction-sections -fdata-sections -nostdlib \
 -Wl,--gc-sections,-T,qd_payload.ld qd_conv.c qd_hmx_ops_encoded.S qd_hmx_native_ops.S -o build/qd_conv_native.elf
'
