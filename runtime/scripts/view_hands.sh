#!/usr/bin/env bash
# Launch the already-built OpenXR hand-joint app on Frame.
# Pass --nir to run the stereo NIR tracking-camera viewer instead of the color camera viewer.
set -euo pipefail
host="${FRAME_HOST:-frame}"
remote_dir="${FRAME_REMOTE_DIR:-}"

script="./scripts/run_hand_viewer.sh"
args=()
is_nir=false
for argument in "$@"; do
    if [[ "$argument" == "--nir" ]]; then
        script="./scripts/run_nir_hand_viewer.sh"
        is_nir=true
    else
        args+=("$argument")
    fi
done

if [ -z "$remote_dir" ]; then
    if [[ "$is_nir" == "true" ]]; then
        if ssh "$host" 'test -d /home/steamos/nirhands-npu/runtime'; then
            remote_dir="/home/steamos/nirhands-npu"
        else
            remote_dir="/home/steamos/frame-spatial-runtime"
        fi
    elif ssh "$host" 'test -d /home/steamos/hexecutor-hand-viewer/runtime'; then
        remote_dir="/home/steamos/hexecutor-hand-viewer"
    else
        remote_dir="/home/steamos/frame-spatial-runtime"
    fi
fi
quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
command="cd $(quote "$remote_dir/runtime") && exec $script"
for argument in "${args[@]}"; do command+=" $(quote "$argument")"; done
exec ssh -t "$host" "$command"
