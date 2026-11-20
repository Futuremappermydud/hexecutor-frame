#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# usage: run.sh src.c args...
cd "$(dirname "$0")/../../.."
src=$1; shift
docker run --rm --platform linux/amd64 -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc "
SDK=/repo/samples/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
\$SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -mhmx -O2 -G0 -w $src hexecutor/native/qd_hmx_ops.S -lhexagon -o /tmp/a.elf && for a in $*; do \$SDK/bin/hexagon-sim -mv75 --quiet /tmp/a.elf -- \$(echo \$a | tr , ' ') 2>&1 | grep -v rev_id; done"
