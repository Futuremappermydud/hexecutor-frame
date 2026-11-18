#!/bin/sh
# Runs a whole lowered Q16P program on the Hexagon simulator with the HMX
# leaf and the HVX tile leaves (hexecutor/tests/test_qnn16_program_sim.c).
# Usage: test_qnn16_program_sim.sh [--compare] [--stop N] [--fuzz KIND [--all]] PREFIX OUT_DIR INPUT.f32...
# --fuzz runs only ops of one kind on random operands, generic path against tile path.
# QNN16_SIM_ELF names the test binary, so two runs can share a tree.
# PREFIX.q16p/.q16c, inputs and outputs are local files under the repository.
set -eu
cd "$(dirname "$0")/../.."
: "${HEXECUTOR_SDK_ROOT:?set HEXECUTOR_SDK_ROOT}"
: "${HEXECUTOR_TOOLCHAIN_IMAGE:?set HEXECUTOR_TOOLCHAIN_IMAGE}"
compare=
emu=
if [ "${1:-}" = --parts4 ] || [ "${1:-}" = --parts4-rev ]; then emu=$1; shift; fi
if [ "${1:-}" = --compare ]; then compare=--compare; shift; fi
if [ "${1:-}" = --stop ]; then compare="$compare --stop $2"; shift 2; fi
if [ "${1:-}" = --fuzz ]; then compare="--fuzz $2"; shift 2; if [ "${1:-}" = --all ]; then compare="$compare --all"; shift; fi; fi
elf=${QNN16_SIM_ELF:-test_program.elf}
prefix=$1 out=$2
shift 2
mkdir -p hexecutor/native/build/qnn16 "$out"
docker run --rm --pull never --platform linux/amd64 \
  -v "$HEXECUTOR_SDK_ROOT:/sdk:ro" -v "$PWD:/workspace" -w /workspace \
  -e QNN16_SIM_FLAGS="${QNN16_SIM_FLAGS:-}" -e QNN16_SIM_CFLAGS="${QNN16_SIM_CFLAGS:-}" -e ELF="$elf" "$HEXECUTOR_TOOLCHAIN_IMAGE" sh -eu -c '
tools=/sdk/tools/HEXAGON_Tools/19.0.07/Tools
q=hexecutor/native/qnn16
$tools/bin/hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128b -O2 -ffp-contract=off -fno-strict-aliasing -I$q \
  ${QNN16_SIM_CFLAGS:-} hexecutor/tests/test_qnn16_program_sim.c $q/program.c $q/exec.c $q/qnn16_f32.c $q/qnn16_hvx.c $q/qnn16_tile*.c \
  $q/qnn16_hmx.S -lhexagon -o hexecutor/native/build/qnn16/$ELF
$tools/bin/hexagon-sim -mv75 --quiet ${QNN16_SIM_FLAGS:-} hexecutor/native/build/qnn16/$ELF -- "$@"
' sh $emu $compare "$prefix.q16p" "$prefix.q16c" "$out" "$@"
