#!/bin/sh
set -eu
cd "$(dirname "$0")/../../../.."
N=hexecutor/native/program
mkdir -p /tmp/planes-host
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -fno-strict-aliasing -w -DNPRG2_APPROX_COST_VOLUME_STEP=0 -DNPRG2_HW_ROUNDING_U16=1 -DNPRG2_PLANES=1 -DONE_INPLACE=1 -DSLIDE_INPLACE_ALL=1 -DCHAIN_LUT=1 -DEW_MEMO=1 "$@" -I$N $N/test_prepared_full_model.c $N/prepared_reference.c $N/prepared_executor.c $N/nprg2_runtime.c $N/prepared_abs.c $N/prepared_addsub_flat.c $N/prepared_block_zap.c $N/prepared_flat_vtcm.c $N/prepared_io.c $N/prepared_mean.c $N/prepared_mul_flat.c $N/prepared_quantize.c $N/prepared_quantize_core.c $N/prepared_reshape_cb.c $N/prepared_sign.c $N/prepared_spill_fill.c $N/prepared_fuse.c $N/prepared_dma_ring.c hexecutor/native/qd_graph.c hexecutor/native/qd_resident.c -lm -pthread -o /tmp/planes-host/lanes_host
