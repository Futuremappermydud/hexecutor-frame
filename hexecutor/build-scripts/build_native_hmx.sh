#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Separate experimental payload. Never rebuild the loader or replace v5.
set -eu
cd "$(dirname "$0")/../.."
task_sdk_root="${HEXECUTOR_SDK_ROOT:-$PWD/samples/depth-model/.work/hexagon-sdk/6.6.0.0}"
docker run --rm --pull never --platform linux/amd64 -v "$task_sdk_root:/sdk:ro" -v "$PWD:/workspace" -w /workspace/hexecutor/native \
 "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -eu -c '
cc=/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang
flags="-I/workspace/samples/depth-model/deployment/generated -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -ffp-contract=off -fno-strict-aliasing -Wall -Wextra"
$cc $flags -DQDC_NATIVE_BUILD=1 -ffunction-sections -fdata-sections -nostdlib \
 -Wl,--gc-sections,-T,qd_payload.ld qd_conv.c qd_hmx_ops_encoded.S qd_hmx_native_ops.S -o build/qd_conv_native.elf
'
