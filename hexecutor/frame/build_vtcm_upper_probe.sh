#!/bin/sh
# Build only the reversible upper-VTCM probe; never replace the loader.
set -eu
cd "$(dirname "$0")/.."
docker run --rm --platform linux/amd64 -v "$PWD/.work/hexagon-sdk/6.6.0.0:/sdk:ro" \
  -v "$PWD/frame:/src" -w /src depth-model-hmx-tools:latest \
  /sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang \
  -mv75 -mhvx -mhvx-length=128b -O2 -G0 -ffreestanding -fno-builtin -fno-stack-protector \
  -Wall -Wextra -nostdlib -Wl,-T,qd_payload.ld \
  bench/vtcm_upper_probe.c -o build/vtcm_upper_probe.elf
