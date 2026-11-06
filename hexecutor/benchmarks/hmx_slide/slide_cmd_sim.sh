#!/bin/sh
# usage (from depth-model/): slide_cmd_sim.sh [build-only]   one container, variant 32 (1x1 packed native path, conv_path 4)
# configs: old (-DNAT_1X1=0), new (default)
SDKV=${SDKV:-/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk}
MAXC=${MAXC:-12}
docker run --rm --platform linux/amd64 -e BUILDONLY="$1" -e MAXC="$MAXC" -v "$SDKV:/repo/depth-model/.work/hexagon-sdk" -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
SDK=/repo/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
N=frame/native_program
build() {
  $SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -G0 -w -ffp-contract=off -fno-fast-math -fno-strict-aliasing \
    -DNPRG2_HW_ROUNDING=1 -DNPRG2_SLIDE3=1 -DWANT_PATH=4 $2 -I$SDK/target/hexagon/include -I$N \
    hexecutor/benchmarks/hmx_slide/slide_cmd_sim.c $N/prepared_executor.c $N/prepared_fuse.c $N/nprg2_runtime.c frame/qd_hmx_ops_encoded.S \
    -lhexagon -lm -o /tmp/sc_$1.elf || exit 1
}
build old "-DNAT_1X1=0"
build new ""
[ -n "$BUILDONLY" ] && exit 0
for v in old new; do
  echo "== $v"
  $SDK/bin/hexagon-sim -mv75 --quiet /tmp/sc_$v.elf -- hexecutor/native/build/resident-primary.qdrx $N/examples/primary-v2.nprg 32 recovered/primary/constants/constant_pool_5.bin $MAXC 2>&1 | grep -v rev_id
done'
