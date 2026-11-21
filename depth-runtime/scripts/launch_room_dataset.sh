#!/usr/bin/env bash
# Steam menu target. A controller trigger+grip hold requests a clean stop.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${FRAME_DATASET_PYTHON:-}
if [[ -z "$PYTHON" ]]; then
    for candidate in "$HOME/frame-dsp/depth-model/.venv/bin/python" "$HOME/frame-dsp/quest-depth/.venv/bin/python"; do
        if [[ -x "$candidate" ]]; then PYTHON=$candidate; break; fi
    done
    PYTHON=${PYTHON:-python3}
fi
OUT_ROOT=${FRAME_DATASET_ROOT:-$ROOT/experiments/room-dataset}
mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/.capture.lock"
if ! flock -n 9; then
    echo 'A room dataset capture is already active.' >&2
    exit 1
fi
SESSION_ID=$(date -u +%Y%m%dT%H%M%SZ)-scan-$$
SESSION="$OUT_ROOT/$SESSION_ID"
mkdir "$SESSION"
LAUNCH_LOG="$OUT_ROOT/$SESSION_ID-launcher.log"
printf '%s\n' "$SESSION_ID" > "$OUT_ROOT/active-session"
cleanup() {
    if [[ -f "$LAUNCH_LOG" ]]; then mv -f "$LAUNCH_LOG" "$SESSION/launcher.log"; fi
    rm -f "$OUT_ROOT/active-session"
}
CAPTURE_PID=
STOP_REQUESTED=0
stop_capture() {
    STOP_REQUESTED=1
    if [[ -n "$CAPTURE_PID" ]]; then kill -TERM "$CAPTURE_PID" 2>/dev/null || true; fi
}
trap stop_capture INT TERM
trap cleanup EXIT
{
    echo "Session: $SESSION"
    echo "Started: $(date -u +%FT%TZ)"
    if [[ $(sysctl -n kernel.yama.ptrace_scope) != 0 ]]; then
        echo 'Capture needs kernel.yama.ptrace_scope=0; run scripts/setup_frame.sh from the Mac after boot.' >&2
        exit 1
    fi
    if ! pgrep -x vrcompositor >/dev/null; then
        echo 'vrcompositor is not running.' >&2
        exit 1
    fi
    if [[ -f "$OUT_ROOT/.npu-fault-boot-id" ]] &&
       [[ $(cat "$OUT_ROOT/.npu-fault-boot-id") == $(cat /proc/sys/kernel/random/boot_id) ]]; then
        echo 'The prior cDSP worker did not clean up. Reboot the Frame before another Model NPU scan.' >&2
        exit 1
    fi
    MODEL_BACKEND=${FRAME_DATASET_MODEL_BACKEND:-prepared-qnn}
    if [[ "$MODEL_BACKEND" == prepared-qnn ]] && [[ ! -e /sys/module/cdsp_mem/parameters/size_mb ]]; then
        echo 'QNN needs the checked 128 MB cdsp_mem lease. Run scripts/setup_frame.sh --qnn from the Mac after boot.' >&2
        exit 1
    fi
    ARGS=(capture "$SESSION" --live-viewer --scene "${FRAME_DATASET_SCENE:-room-main}"
          --split "${FRAME_DATASET_SPLIT:-development}"
          --model-backend "$MODEL_BACKEND"
          --frames "${FRAME_DATASET_FRAMES:-0}"
          --spacing-ms "${FRAME_DATASET_SPACING_MS:-0}"
          --min-free-gb "${FRAME_DATASET_MIN_FREE_GB:-8}"
          --ir --ir-max-pairs "${FRAME_DATASET_IR_MAX_PAIRS:-0}"
          --notes 'Launched from headset menu')
    if [[ ${FRAME_DATASET_NO_DOTS:-0} != 1 ]]; then ARGS+=(--dots); fi
    if [[ ${FRAME_DATASET_NO_MARKERS:-0} == 1 ]]; then ARGS+=(--no-markers); fi
    if [[ ${FRAME_DATASET_KEEP_FLOODS:-0} == 1 ]]; then ARGS+=(--keep-floods); fi
    if (( STOP_REQUESTED )); then exit 130; fi
    "$PYTHON" "$ROOT/scripts/room_dataset.py" "${ARGS[@]}" &
    CAPTURE_PID=$!
    if (( STOP_REQUESTED )); then stop_capture; fi
    while :; do
        status=0
        wait "$CAPTURE_PID" || status=$?
        if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then break; fi
    done
    if (( status != 0 )); then exit "$status"; fi
    echo "Capture finished: $(date -u +%FT%TZ)"
    echo "Pull from Mac: python3 scripts/pull_room_dataset.py --session $SESSION_ID"
} > "$LAUNCH_LOG" 2>&1
