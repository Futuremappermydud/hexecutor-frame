#!/bin/sh
set -eu
[ "$#" -eq 1 ] || { echo "usage: $0 WORK_DIRECTORY" >&2; exit 2; }
BIN=$(cd "$1" && pwd)/bin
mkdir -p "$BIN"
cd "$1/src"; N=frame/native_program
cc_one(){ n=$1; shift; cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -w "$@" -I$N $N/test_prepared_full_model.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c $N/prepared_abs.c $N/prepared_addsub_flat.c $N/prepared_block_zap.c $N/prepared_flat_vtcm.c $N/prepared_io.c $N/prepared_mean.c $N/prepared_mul_flat.c $N/prepared_quantize.c $N/prepared_quantize_core.c $N/prepared_reshape_cb.c $N/prepared_sign.c $N/prepared_spill_fill.c $N/prepared_fuse.c frame/qd_graph.c frame/qd_resident.c -lm -pthread -o "$BIN/test_$n" >"$BIN/build_$n.log" 2>&1 || { cat "$BIN/build_$n.log" >&2; return 1; }; echo "built $n"; }
cc_one emu -DNPRG2_APPROX_COST_VOLUME_STEP=0
cc_one hw -DNPRG2_APPROX_COST_VOLUME_STEP=0 -DNPRG2_HW_ROUNDING=1
cc_one fix -DNPRG2_APPROX_COST_VOLUME_STEP=0 -DNPRG2_HW_ROUNDING=1 -DNPRG2_HW_BIAS_FIX=1
cc_one mixed -DNPRG2_APPROX_COST_VOLUME_STEP=0 -DNPRG2_HW_ROUNDING_U16=1
