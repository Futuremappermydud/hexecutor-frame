#!/bin/sh
# usage: run.sh src.c args...
cd "$(dirname "$0")/../../.."
src=$1; shift
docker run --rm --platform linux/amd64 -v "$PWD:/repo" -w /repo depth-model-hmx-tools:latest sh -lc "
SDK=/repo/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
\$SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -mhmx -O2 -G0 -w $src frame/qd_hmx_ops.S -lhexagon -o /tmp/a.elf && for a in $*; do \$SDK/bin/hexagon-sim -mv75 --quiet /tmp/a.elf -- \$(echo \$a | tr , ' ') 2>&1 | grep -v rev_id; done"
