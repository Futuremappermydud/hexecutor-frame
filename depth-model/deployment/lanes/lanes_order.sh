#!/bin/sh
# usage: lanes_order.sh BIN EVENTS_DIR : replay adversarial topological orders on the host against the oracle.
BIN=$1; EV=$2
cd "$(dirname "$0")/../../.."
H=hexecutor/native/build/ab/investigation-stream-real-20261105/host_hashes.json
run(){ c=$1; dir=$2; o=$3; hs=$(python3 -c "import json;d=json.load(open('$H'));print(','.join(d['$c']))")
  r=$(QD_HMX=1 QD_EXPECT=$hs QD_INPUTS=$dir QD_RUNLIST_EVENTS=$EV/$o.bin $BIN primary fast runlist-main 2>&1; echo rc=$?); echo "$c $o: $(echo "$r" | tail -2 | cut -c1-150 | tr '\n' ' ')"; }
for o in ${ORDERS:-sched earliest latest random1 random2 random3 mainfirst workerfirst}; do
  run smoke hexecutor/native/build/native-smoke $o &
  run random hexecutor/native/build/depth-inputs/random $o &
  run h+8 /tmp/frame-rounding-ablation/inputs/h+8 $o &
  wait
done
