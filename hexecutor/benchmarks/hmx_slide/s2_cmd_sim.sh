#!/bin/sh
# usage (from depth-model/): s2_cmd_sim.sh   one container: builds old (S2_FORCE_BCOPY=1) and new, runs variant 36 conv commands
SDKV=${SDKV:-/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/hexagon-sdk}
MAXC=${MAXC:-12}
docker run --rm --platform linux/amd64 -e MAXC="$MAXC" -v "$SDKV:/repo/depth-model/.work/hexagon-sdk" -v "$PWD:/repo" -w /repo "${HEXECUTOR_TOOLCHAIN_IMAGE:-hexecutor-toolchain:local}" sh -lc '
SDK=/repo/depth-model/.work/hexagon-sdk/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
N=frame/native_program
build() { # name flags
  $SDK/bin/hexagon-clang -mv75 -mhvx -mhvx-length=128b -O2 -G0 -w -ffp-contract=off -fno-fast-math -fno-strict-aliasing \
    -DNPRG2_HW_ROUNDING=1 -DNPRG2_SLIDE3=1 -DS2_COUNT=1 $2 -I$SDK/target/hexagon/include -I$N \
    hexecutor/benchmarks/hmx_slide/s2_cmd_sim.c $N/prepared_executor.c $N/prepared_fuse.c $N/nprg2_runtime.c frame/qd_hmx_ops_encoded.S \
    -lhexagon -lm -o /tmp/s2c_$1.elf || exit 1
}
build old "-DS2_FORCE_BCOPY=1"
build new ""
for v in old new; do
  echo "== $v"
  $SDK/bin/hexagon-sim -mv75 --quiet /tmp/s2c_$v.elf -- hexecutor/native/build/resident-primary.qdrx $N/examples/primary-v2.nprg 36 recovered/primary/constants/constant_pool_5.bin $MAXC 4 2>&1 | grep -v rev_id
done'
