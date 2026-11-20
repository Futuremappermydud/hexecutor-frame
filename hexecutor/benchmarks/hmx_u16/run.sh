#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# usage (from depth-model/): hexecutor/benchmarks/hmx_u16/run.sh "label:-DN16_PT=0 -DN16_DIRECT_OUT=0" "label2:-D..." ...
# One container, one build + run per variant. Prints per-config hash/cycles.
cd "$(dirname "$0")/../../.."
SPEC="$*"
docker run --rm --platform linux/amd64 -v "$PWD:/repo" -v /Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk:/repo/depth-model/.work/hexagon-sdk -w /repo -e SPEC="$SPEC" -e BUILD_ONLY="$BUILD_ONLY" "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -c '
SDK=/repo/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
echo "$SPEC" | tr "|" "\n" | while IFS= read -r v; do
 L=${v%%:*}; F=${v#*:}
 $SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -mhmx -O2 -G0 -w -ffp-contract=off -fno-fast-math -fno-strict-aliasing -DNPRG2_HW_ROUNDING=1 -DNPRG2_SLIDE3=1 $F -I$SDK/target/hexagon/include -Ihexecutor/native/program hexecutor/benchmarks/hmx_u16/n16_test.c hexecutor/native/qd_hmx_ops.S hexecutor/native/program/prepared_executor.c hexecutor/native/program/prepared_fuse.c hexecutor/native/program/nprg2_runtime.c -lhexagon -lm -o /tmp/n16_$L.elf || { echo BUILD FAIL $L; continue; }
 [ -n "$BUILD_ONLY" ] && continue
 echo "== $L ($F)"
 $SDK/bin/hexagon-sim -mv75 --quiet /tmp/n16_$L.elf 2>&1 | grep -v rev_id
done'
