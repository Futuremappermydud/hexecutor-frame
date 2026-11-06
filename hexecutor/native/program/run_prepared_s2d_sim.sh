#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."
if [ ! -x .venv-cpu/bin/python ]; then
  echo "Missing .venv-cpu/bin/python; create the project CPU environment to capture CPU-prefix fixtures." >&2
  exit 1
fi
.venv-cpu/bin/python hexecutor/native/program/capture_s2d_cpu_prefix.py --case smoke --model primary --model alternate
docker run --rm --platform linux/amd64 -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
  SDK=/repo/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
  ELF=/tmp/nprg2-prepared-s2d.elf
  "$SDK/bin/hexagon-clang" -mv75 -mhvx -mhvx-length=128b -O2 -G0 -Wall -Wextra -Werror \
    -I"$SDK/target/hexagon/include" -Ihexecutor/native/program -Ihexecutor/native \
    hexecutor/native/program/test_prepared_s2d.c hexecutor/native/program/prepared_s2d.c \
    hexecutor/native/program/prepared_executor.c hexecutor/native/program/nprg2_runtime.c -o "$ELF"
  "$SDK/bin/hexagon-sim" -mv75 --quiet "$ELF"
'
