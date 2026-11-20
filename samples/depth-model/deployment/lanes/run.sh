#!/usr/bin/env bash
# Offline device benchmark: run.sh NAME REPEAT TAG [device_check options...]
set -euo pipefail
name=${1:?engine name required}; repeat=${2:?repeat count required}; tag=${3:?report tag required}; shift 3
[[ $name =~ ^[A-Za-z0-9_-]+$ && $tag =~ ^[A-Za-z0-9_-]+$ && $repeat =~ ^[0-9]+$ ]] || { echo 'invalid name, tag, or repeat count' >&2; exit 2; }
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
build=$repo/hexecutor/native/build/ab/planes-20261105
host=${FRAME_DEVICE_HOST:-frame}
remote=${FRAME_REMOTE_ROOT:-frame-spatial-runtime}
python=${FRAME_OFFLINE_PYTHON:-python3}
report=depth-runtime/experiments/ab/planes-20261105/$name-$tag.json
remote_build=hexecutor/native/build/ab/planes-20261105
quote() { printf '%q' "$1"; }
mkdir -p /tmp/planes-host
rm -f "$build/$name-$tag.json"
ssh -o ConnectTimeout=8 "$host" "mkdir -p $(quote "$remote/$remote_build") $(quote "$remote/depth-runtime/experiments/ab/planes-20261105"); rm -f $(quote "$remote/$report")"
rsync -c "$build/eng-$name.elf" "$build/boot-$name.elf" "$host:$remote/$remote_build/"
command="cd $(quote "$remote") && PYTHONPATH=.:samples/depth-model/python:samples/depth-model/tools:hexecutor/benchmarks $(quote "$python") hexecutor/native/build/ab/investigation-stream-real-20261105/device_check.py"
for arg in --model primary --repeat "$repeat" "$@" --engine "$remote_build/eng-$name.elf" --bootstrap "$remote_build/boot-$name.elf" --prepared-program samples/depth-model/deployment/programs/primary-v2.nprg --report "$report"; do
    command+=" $(quote "$arg")"
done
# The diagnostic script and private fixtures must already be provisioned on the device.
status=0
timeout 240 ssh "$host" "$command" > "/tmp/planes-host/$name-$tag.log" 2>&1 || status=$?
rsync -c "$host:$remote/$report" "$build/$name-$tag.json"
python3 - "$build/$name-$tag.json" <<'PY'
import json, statistics, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
runs = result['runs']
timings = [x['dsp_ms'] for x in runs]
print('passed', result['passed'], 'runs', len(runs), 'exit_errors', result.get('exit_errors'))
if timings:
    print('first ms', timings[0], 'warm median ms', statistics.median(timings[1:] or timings))
print('failed cases', [(x['case'], x['run']) for x in runs if not x['passed']])
PY
exit "$status"
