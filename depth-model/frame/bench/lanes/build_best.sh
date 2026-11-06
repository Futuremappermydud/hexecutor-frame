#!/bin/sh
# Build the current best primary engine. usage: build_best.sh NAME ["extra flags, + separated"]
# Output: frame/build/ab/planes-20261105/{eng,boot}-NAME.elf. Regenerate native_program/qnn_lanes_primary.generated.h
# with gen_qnn_lanes.py and the options in LANE_OPTS when the program or the schedule policy changes.
# LANE_OPTS: --workers 2 --costs bench/lanes/cmdcost-l7.json --drop 1/10 --move 26/8,3/1,4/28,4/27,4/29
#            --cpu-weights 4/28,4/27,4/29 --free-main --main-steal-us 6 --active block
set -eu
cd "$(dirname "$0")/../../.."
F="-DNPRG2_PLANES=1+-DQDS_POOL1_NOCOPY=1+-DONE_INPLACE=1+-DQDS_VTCM_SINGLE_PAGE=1+-DSLIDE_INPLACE_ALL=1+-DSETUP_FAST=1+-DCHAIN_LUT=1+-DEW_MEMO=1"
F="$F+-DQDS_QNN_LANES=1+-DQDS_L2_SIG=1+-DPLANE_SAT=1+-DSLIDE_TRIM=1+-DSLIDE_TRIM2=1+-DWEIGHTS_MEMO=1+-DWEIGHTS_CPU_BELOW=2048"
F="$F+-DWEIGHTS_CPU_VARIANTS=0x38000000u+-DQDS_HMX_STAY=1+-DCMD_MEMO=1+-DCHAIN_HVX=1+-mhvx-ieee-fp"
F="$F+-DQDS_LANE_PAUSE=255+-DCLAIM_NOBAR=1+-DLANE_BAR=1+-DLANE_PUBLISH=1"
PROFILE=${PROFILE:-0} sh frame/bench/planes_build.sh "$1" "$F${2:++$2}"
