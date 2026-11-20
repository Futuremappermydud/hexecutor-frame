#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/../../.."
docker run --rm --platform linux/amd64 -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
 SDK=/repo/samples/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
 ELF=/tmp/nprg2-mul-flat.elf
 "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 -Wall -Wextra -Werror -I"$SDK/target/hexagon/include" -Ihexecutor/native/program hexecutor/native/program/test_prepared_mul_flat.c hexecutor/native/program/prepared_mul_flat.c hexecutor/native/program/prepared_executor.c hexecutor/native/program/nprg2_runtime.c -o "$ELF"
 "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
'
