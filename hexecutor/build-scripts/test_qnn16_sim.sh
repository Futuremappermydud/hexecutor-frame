#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Hexagon simulator checks of the QNN FP16 run-list handlers (hexecutor/native/qnn16)
# against fixtures from hexecutor/offline/qnn16_oracle.py. Fixtures are local,
# ignored artifacts: pass their directory as QNN16_FIXTURES (default
# samples/hand-tracking/.work/qnn16-fixtures, relative to the repository root).
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE}"
fixtures="${QNN16_FIXTURES:-hand-tracking/.work/qnn16-fixtures}"
set -- "$fixtures"/*-conv1x1*.bin
[ -f "$1" ] || { echo "no conv1x1 fixtures in $fixtures" >&2; exit 1; }
mkdir -p hexecutor/native/build/qnn16
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
$tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -Ihexecutor/native/qnn16 \
  hexecutor/tests/test_qnn16_conv_sim.c hexecutor/native/qnn16/qnn16_hmx.S \
  -o hexecutor/native/build/qnn16/test_conv.elf
$tools/bin/hexagon-sim -mv75 --quiet hexecutor/native/build/qnn16/test_conv.elf -- "$@" \
  > hexecutor/native/build/qnn16/test_conv.log
cat hexecutor/native/build/qnn16/test_conv.log
grep -q "^PASS qnn16 conv1x1" hexecutor/native/build/qnn16/test_conv.log
' sh "$@"
