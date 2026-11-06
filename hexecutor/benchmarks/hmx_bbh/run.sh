#!/bin/sh
# usage (any cwd): run.sh src.c [arg,arg ...]  one container; args per run comma-separated
cd "$(dirname "$0")/../../.."
SDKV=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk
src=$1; shift
docker run --rm --platform linux/amd64 -v "$SDKV:/sdkv:ro" -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc "
SDK=/sdkv/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
\$SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -mhmx -O2 -G0 -w $src hexecutor/native/qd_hmx_ops.S -lhexagon -o /tmp/a.elf || exit 1
for a in ${*:-0}; do timeout 300 \$SDK/bin/hexagon-sim -mv75 --quiet /tmp/a.elf -- \$(echo \$a | tr , ' ') 2>&1 | grep -v rev_id; done"
