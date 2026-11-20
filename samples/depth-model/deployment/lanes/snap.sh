#!/bin/sh
# usage: snap.sh NAME : sources-only snapshot of the candidate
set -eu
cd "$(dirname "$0")/../../../.."
OUT=hexecutor/native/build/ab/planes-20261105
mkdir -p "$OUT"
find hexecutor samples/depth-model/python samples/depth-model/deployment -type f   \( -name '*.c' -o -name '*.h' -o -name '*.S' -o -name '*.inc' -o -name '*.py' -o -name '*.sh' -o -name Kbuild \)   -not -path '*/build/*' -not -path '*/generated/*' -size -2000k | tar czf "$OUT/source-$1.tar.gz" -T -
ls -l "$OUT/source-$1.tar.gz"
