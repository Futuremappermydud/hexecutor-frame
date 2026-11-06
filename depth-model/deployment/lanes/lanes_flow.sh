#!/bin/sh
# usage: lanes_flow.sh TAG "generator options" "extra cflags" : generate, host-replay, build prod+profile
set -eu
TAG=$1; OPTS=$2; EXTRA=${3:-}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
LANES=$REPO/depth-model/deployment/lanes
cd "$REPO"
R=${LANE_RUNLIST:-$REPO/hexecutor/native/build/ab/planes-20261105/nomod-v2.json}
PYTHONDONTWRITEBYTECODE=1 python3 "$LANES/gen_qnn_lanes.py" "$R" depth-model/deployment/generated/qnn_lanes_primary.generated.h --workers 2 --costs ${COSTS:-/tmp/planes-host/costs/cmdcost.json} $OPTS --events /tmp/planes-host/ev-$TAG 2>&1 | grep -E 'modeled|critical|rror' | cut -c1-120
ORDERS="sched latest random1 random2 mainfirst workerfirst" "$LANES/lanes_order.sh" /tmp/planes-host/lanes_host /tmp/planes-host/ev-$TAG 2>&1 | grep -vE 'both graph outputs match' | cut -c1-200
echo "host replay done"
cd "$REPO"
F="-DNPRG2_PLANES=1+-DQDS_POOL1_NOCOPY=1+-DONE_INPLACE=1+-DQDS_VTCM_SINGLE_PAGE=1+-DSLIDE_INPLACE_ALL=1+-DSETUP_FAST=1+-DCHAIN_LUT=1+-DEW_MEMO=1+-DQDS_QNN_LANES=1${EXTRA:++$EXTRA}"
PROFILE=0 sh hexecutor/benchmarks/planes_build.sh $TAG "$F" 2>&1 | grep -E ' error|eng-.*elf$' | cut -c1-80
