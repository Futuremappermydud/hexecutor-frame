#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Build only the reversible upper-VTCM probe; never replace the loader.
set -eu
cd "$(dirname "$0")/../.."
docker run --rm --pull never --platform linux/amd64 -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD:/workspace" -w /workspace/hexecutor/native "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -fno-stack-protector \
  -Wall -Wextra -nostdlib -Wl,-T,qd_payload.ld \
  ../benchmarks/vtcm_upper_probe.c -o build/vtcm_upper_probe.elf
