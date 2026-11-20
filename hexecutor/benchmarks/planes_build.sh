#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# usage: planes_build.sh NAME "extra cflags (+ separated)"   -> hexecutor/native/build/ab/planes-20261105/{eng,boot}-NAME.elf in the MAIN tree
set -eu
cd "$(dirname "$0")/../.."
OUT=$PWD/hexecutor/native/build/ab/planes-20261105
mkdir -p "$OUT"
SDKV=$PWD/depth-model/.work/hexagon-sdk
docker run --rm --pull never --platform linux/amd64 -v "$SDKV:/sdkv:ro" -v "$PWD:/workspace" -w /workspace/hexecutor/native "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -c '/sdkv/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-clang -mv75 -mhmx -c qd_hmx_ops.S -o build/ops.o'
python3 hexecutor/benchmarks/encode_hmx.py hexecutor/native/build/ops.o --source hexecutor/native/qd_hmx_ops.S --output hexecutor/native/qd_hmx_ops_encoded.S
n=$1; extra=${2:-}
log=$OUT/build-$n.log
env QDS_BLOCKED_GLUE=1 QDS_DYNAMIC_TCM=1 QDS_FALLBACK_TCM=2 QDS_FAST_SPILL=1 QDS_NATIVE_1X1=1 QDS_NATIVE_3X3=1 QDS_NATIVE_DIRECT_RANGE=1 QDS_NATIVE_PREPARED=1 \
    QDS_PREPARED_ELEMENTWISE=1 QDS_PREPARED_REFERENCE=1 QDS_PRESERVE_VTCM=1 QDS_VECTORIZE=0 QDS_PREPARED_PROFILE=${PROFILE:-1} QDS_PREPARED_THREADS=4 QDS_PREPARED_FUSE=0 \
    QDS_PREPARED_HMX=1 QDS_L2_CACHE=1 QDS_PREPARED_DMA_ASYNC=1 \
    QDS_EXTRA_CFLAGS="-DNPRG2_HW_ROUNDING_U16=1+-DNPRG2_SLIDE3=1+-DNPRG2_QNN_RUNLISTS=1+-DQDS_QNN_CLAIM=1+-DQDS_PREPARED_DMA_MODE=0+-O3${extra:++$extra}" \
    sh hexecutor/build-scripts/build_resident_dsp.sh >"$log" 2>&1 || { tail -30 "$log"; exit 1; }
eng=hexecutor/native/$(sed -n 's/^Experimental engine: \([^;]*\);.*/\1/p' "$log")
boot=hexecutor/native/$(sed -n 's/.*bootstrap: \(.*\)$/\1/p' "$log")
cp "$eng" "$OUT/eng-$n.elf"; cp "$boot" "$OUT/boot-$n.elf"
shasum -a 256 "$OUT/eng-$n.elf" "$OUT/boot-$n.elf"
grep -E "error|warning: (implicit|incompatible)" "$log" | head -5 || true
