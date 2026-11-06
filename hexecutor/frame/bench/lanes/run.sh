#!/bin/sh
# usage: run.sh NAME REPEAT [--all-cases | --case smoke ...]  ; report results/ab/planes-20261105/NAME-TAG.json
set -u
n=$1; rep=$2; tag=$3; shift 3
M=/Users/owenelliott/Documents/projects/spatial-stuff/depth-model; R=frame-dsp/depth-model/wave1-20261101
# Never report an earlier run's file as this run's result.
rm -f $M/frame/build/ab/planes-20261105/$n-$tag.json
ssh -o ConnectTimeout=8 frame true || { echo "no report: frame unreachable"; exit 1; }
ssh frame "rm -f $R/results/ab/planes-20261105/$n-$tag.json"
ssh frame "mkdir -p $R/frame/build/ab/planes-20261105 $R/results/ab/planes-20261105"
rsync -c $M/frame/build/ab/planes-20261105/eng-$n.elf $M/frame/build/ab/planes-20261105/boot-$n.elf frame:$R/frame/build/ab/planes-20261105/
timeout 240 ssh frame "cd $R && /home/steamos/frame-dsp/depth-model/.venv/bin/python frame/build/ab/investigation-stream-real-20261105/device_check.py --model primary --repeat $rep $* --engine frame/build/ab/planes-20261105/eng-$n.elf --bootstrap frame/build/ab/planes-20261105/boot-$n.elf --prepared-program frame/native_program/examples/primary-v2.nprg --report results/ab/planes-20261105/$n-$tag.json" > /tmp/planes-host/$n-$tag.log 2>&1
echo "exit $?"
rsync -c frame:$R/results/ab/planes-20261105/$n-$tag.json $M/frame/build/ab/planes-20261105/ 2>/dev/null
python3 - "$M/frame/build/ab/planes-20261105/$n-$tag.json" <<'P'
import json,sys,statistics as st
try: d=json.load(open(sys.argv[1]))
except Exception as e: print('no report',e); sys.exit()
r=d['runs']; print('passed',d['passed'],'n',len(r),'exit_errors',d.get('exit_errors'),'vtcm',str(d.get('vtcm_lease'))[:100])
ms=[x['dsp_ms'] for x in r]; print('first',round(ms[0],2),'warm median',round(st.median(ms[1:] or ms),3),'min',round(min(ms),3),'max',round(max(ms[1:] or ms),3))
print('fail',[ (x['case'],x['run']) for x in r if not x['passed']][:10])
x=r[-1]; print('phases',{k:round(v,2) for k,v in x['near_conv_phases_ms'].items()},'exec',{k:round(v,2) if isinstance(v,float) else v for k,v in x['executor_profile'].items()})
hp=sorted(x['handler_profile'],key=lambda h:-h['run_ms'])[:8]; print([(h['opcode'],h['variant'],round(h['run_ms'],2)) for h in hp])
wk=sorted(x['handler_profile'],key=lambda h:-h['validate_ms'])[:6]; print('worker',round(sum(h['validate_ms'] for h in x['handler_profile']),2),[(h['opcode'],h['variant'],round(h['validate_ms'],2)) for h in wk])
P
