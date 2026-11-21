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

quote() { printf "'%s'" "${1//\'/\\\'\'}"; }
runtime_candidates=()
if [[ -n "$remote_dir" ]]; then
    runtime_candidates+=("$remote_dir/depth-runtime" "$remote_dir/runtime")
elif [[ "$is_nir" == "true" ]]; then
    runtime_candidates+=(/home/steamos/nirhands-npu/{depth-runtime,runtime}
                        /home/steamos/frame-spatial-runtime/{depth-runtime,runtime})
else
    runtime_candidates+=(/home/steamos/frame-spatial-runtime/depth-runtime
                        /home/steamos/hexecutor-hand-viewer/{depth-runtime,runtime}
                        /home/steamos/frame-spatial-runtime/runtime)
fi
viewer=frame_hand_viewer
if [[ "$is_nir" == "true" ]]; then viewer=frame_nir_hand_viewer; fi
command="for runtime in"
for candidate in "${runtime_candidates[@]}"; do command+=" $(quote "$candidate")"; done
command+="; do if test -x \"\$runtime/$script\" && test -x \"\$runtime/bin/$viewer\"; then cd \"\$runtime\" && exec $script"
for argument in ${args[@]+"${args[@]}"}; do command+=" $(quote "$argument")"; done
command+="; fi; done; echo 'No installed $viewer found on Frame.' >&2; exit 1"
exec ssh -t "$host" "$command"
