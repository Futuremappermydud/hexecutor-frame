#!/usr/bin/env bash
# Phase-1 CDSP broker gate (docs/hexecutor/broker.md), run from the development
# machine. It syncs this tree's source to the Frame, builds the native host there,
# and on one test frame compares:
#   1. frame_hand_tracker with its own Session   (direct)
#   2. frame_hand_tracker through hexecutord      (broker)
# and, when a lowered FeatureNN graph exists locally, times it on the DSP.
#
# It does not change device configuration. It stops if the DSP prerequisites
# (ptrace scope, cdsp_mem, XRService controller FeatureNN off the CDSP) are not
# already in place; see depth-runtime/scripts/setup_frame.sh and docs/hexecutor/operations.md.
#
# Usage: depth-runtime/scripts/broker_gate.sh FRAME.ppm [REPEAT]
set -euo pipefail
FRAME_PPM="${1:?usage: broker_gate.sh FRAME.ppm [REPEAT]}"
REPEAT="${2:-100}"
HOST="${FRAME_HOST:-frame}"
REMOTE="${FRAME_BROKER_DIR:-/home/steamos/hexecutor-broker}"
MODELS="${FRAME_HAND_MODELS:-/home/steamos/hexecutor-hand-viewer/models/hand-tracking}"
DSP="${FRAME_HAND_DSP:-/home/steamos/hexecutor-hand-viewer/dsp}"
LOADER="${FRAME_HAND_LOADER:-/home/steamos/frame-dsp/quest-depth/wave1-20261101/frame/build/qd_loader.elf}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${BROKER_GATE_OUT:-$ROOT/depth-runtime/experiments/broker-gate/$(date +%Y%m%d-%H%M%S)}"
FEATURENN="$ROOT/samples/controller-tracking/.work/lowered/featurenn"
FEATURENN_INPUT="$ROOT/samples/controller-tracking/.work/sim/crop0.in.f32"
mkdir -p "$OUT"

echo "[gate] syncing source to $HOST:$REMOTE"
rsync -a --delete \
    --exclude '.git' --exclude 'build/' --exclude '.work/' --exclude 'models/' \
    --exclude 'depth-runtime/bin/' --exclude 'depth-runtime/experiments/' --exclude '*.o' --exclude '*.d' \
    --include 'CMakeLists.txt' --include 'hexecutor/***' --include 'samples/' \
    --include 'samples/hand-tracking/***' \
    --include 'samples/depth-model/' --include 'samples/depth-model/deployment/' \
    --include 'samples/depth-model/deployment/cpp/***' --include 'samples/depth-model/include/***' \
    --include 'depth-runtime/' --include 'depth-runtime/include/***' --include 'depth-runtime/src/***' \
    --include 'depth-runtime/tools/***' --exclude '*' \
    "$ROOT/" "$HOST:$REMOTE/"
scp -q "$FRAME_PPM" "$HOST:$REMOTE/gate-frame.ppm"
HAVE_FNN=0
if [[ -f "$FEATURENN.hxfg" && -f "$FEATURENN_INPUT" ]]; then
    scp -q "$FEATURENN.hxfg" "$FEATURENN.bin" "$FEATURENN.json" "$FEATURENN_INPUT" "$HOST:$REMOTE/"
    HAVE_FNN=1
fi

ssh "$HOST" bash -s -- "$REMOTE" "$MODELS" "$DSP" "$LOADER" "$REPEAT" "$HAVE_FNN" <<'REMOTE_SCRIPT'
set -euo pipefail
REMOTE=$1 MODELS=$2 DSP=$3 LOADER=$4 REPEAT=$5 HAVE_FNN=$6
cd "$REMOTE"
fail() { echo "[gate] STOP: $*" >&2; exit 3; }
[[ -e /dev/cdsp_mem && "$(cat /sys/module/cdsp_mem/parameters/size_mb)" -ge 128 ]] ||
    fail "cdsp_mem 128 MB lease is not loaded (depth-runtime/scripts/setup_frame.sh --qnn)"
S=/home/steamos/.config/openvr/config/cv/xrservice/XRServiceSettings.json
grep -q '"enableForLogic": *false' "$S" || fail "XRService controller FeatureNN may use the CDSP ($S)"
if pgrep -f 'bin/frame_hand_viewer|bin/hexecutord|frame_depth_worker' >/dev/null; then
    fail "another DSP client is running"
fi
STUCK=$(journalctl -b --no-pager | grep -c 'Stereo FeatureNN on cDSP timed out' || true)
[[ "$STUCK" == 0 ]] || fail "XRService already hit CDSP FeatureNN timeouts this boot ($STUCK); reboot first"

echo "[gate] building on $(hostname)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFRAME_BUILD_TESTS=ON >build.log 2>&1
cmake --build build --parallel 4 --target hexecutord frame_hand_tracker hexecutor_tensor_run test_broker >>build.log 2>&1 ||
    { tail -30 build.log; fail "build failed"; }
./build/test_broker
ART=(--loader "$LOADER" --engine "$DSP/engine-planar.elf" --bootstrap "$DSP/bootstrap4.elf" --chunk-probe "$DSP/chunk_probe.elf")

echo "[gate] direct session, $REPEAT frames"
./build/bin/frame_hand_tracker --models "$MODELS" "${ART[@]}" --repeat "$REPEAT" gate-frame.ppm >direct.jsonl 2>direct.err ||
    { tail -5 direct.err; fail "direct run failed"; }

echo "[gate] through hexecutord, $REPEAT frames"
SOCK=/tmp/hexecutord-gate.sock
./build/bin/hexecutord --socket "$SOCK" "${ART[@]}" \
    --graph "palm=$MODELS/palm.hxfg,$MODELS/palm.bin" \
    --graph "landmark=$MODELS/landmark.hxfg,$MODELS/landmark.bin" 2>hexecutord.err &
DAEMON=$!
for i in $(seq 1 50); do [[ -S "$SOCK" ]] && break; sleep 0.1; done
[[ -S "$SOCK" ]] || { kill $DAEMON 2>/dev/null; cat hexecutord.err; fail "daemon did not start"; }
RC=0
./build/bin/frame_hand_tracker --broker "$SOCK" --repeat "$REPEAT" gate-frame.ppm >broker.jsonl 2>broker.err || RC=$?
sleep 3  # past the 2 s idle close, so the log shows the session release
kill -TERM $DAEMON; wait $DAEMON || true
cat hexecutord.err
[[ $RC == 0 ]] || { tail -5 broker.err; fail "broker run failed"; }

if [[ "$HAVE_FNN" == 1 ]]; then
    echo "[gate] FeatureNN on the DSP"
    HEXECUTOR_RUNS=50 ./build/bin/hexecutor_tensor_run featurenn.hxfg featurenn.bin crop0.in.f32 fnn_ \
        "$LOADER" "$DSP/engine-planar.elf" "$DSP/bootstrap4.elf" "$DSP/chunk_probe.elf" \
        >featurenn.out 2>&1 || { tail -5 featurenn.out; fail "FeatureNN run failed"; }
    tail -3 featurenn.out
fi
echo "[gate] DSP state: $(journalctl -b --no-pager | grep -c 'Stereo FeatureNN on cDSP timed out' || true) XRService FeatureNN timeouts this boot"
REMOTE_SCRIPT

for f in direct.jsonl broker.jsonl hexecutord.err; do scp -q "$HOST:$REMOTE/$f" "$OUT/"; done
if [[ "$HAVE_FNN" == 1 ]]; then
    for f in featurenn.out fnn_0.f32 fnn_1.f32; do scp -q "$HOST:$REMOTE/$f" "$OUT/"; done
fi
python3 - "$OUT" <<'PY'
import json, statistics, sys
from pathlib import Path
out = Path(sys.argv[1])
def load(name):
    return [json.loads(l) for l in (out / name).read_text().splitlines() if l.startswith("{")]
d, b = load("direct.jsonl"), load("broker.jsonl")
def summary(rows):
    ms = sorted(r["ms"] for r in rows[1:])  # first frame includes session open / detector warm-up
    return statistics.mean(ms), ms[int(0.95 * (len(ms) - 1))], ms[-1]
for name, rows in (("direct", d), ("broker", b)):
    m, p95, worst = summary(rows)
    print(f"{name:7s} frames {len(rows):4d}  mean {m:7.3f} ms  p95 {p95:7.3f}  max {worst:7.3f}")
same = all(x["hands"] == y["hands"] for x, y in zip(d, b)) and len(d) == len(b)
print("hand outputs identical:", same)
fnn = out / "featurenn.out"
if fnn.exists():
    runs = [l for l in fnn.read_text().splitlines() if l.startswith("run ")]
    dsp = [float(l.split("dsp_ms ")[1]) for l in runs[1:]]
    host = [float(l.split()[3]) for l in runs[1:]]
    if dsp:
        print(f"FeatureNN dsp mean {statistics.mean(dsp):.3f} ms, host mean {statistics.mean(host):.3f} ms ({len(dsp)} warm runs)")
print("results in", out)
PY
