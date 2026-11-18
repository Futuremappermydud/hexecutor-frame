#!/usr/bin/env bash
# Run on Frame from a terminal with a TTY so sudo can prompt for the capture helper.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

capture="${FRAME_NIR_CAPTURE:-/tmp/nirhands-live/stream_nir_pairs}"
if [[ ! -x "$capture" && -x bin/stream_nir_pairs ]]; then
    capture="bin/stream_nir_pairs"
fi
if [[ ! -x "$capture" ]]; then echo "NIR capture helper missing: $capture" >&2; exit 1; fi
if [[ ! -x bin/frame_nir_hand_viewer ]]; then echo 'Build with make nir-hand-viewer first' >&2; exit 1; fi

has_arg() {
    local target="$1"
    shift
    for a in "$@"; do
        if [[ "$a" == "$target" ]]; then return 0; fi
    done
    return 1
}

extra_args=()
DEFAULT_MODELS="${FRAME_NIR_MODELS:-/home/steamos/nirhands-viewer/models}"
DEFAULT_QNN16_DIR="${FRAME_NIR_QNN16_DIR:-/home/steamos/nirhands-npu/run/v4}"
DEFAULT_LOADER="${FRAME_NIR_LOADER:-${FRAME_DSP_LOADER:-/home/steamos/hexecutor-hand-viewer/dsp/qd_loader.elf}}"
DEFAULT_ENGINE="${FRAME_NIR_ENGINE:-/home/steamos/nirhands-npu/run/v4/engine-noprof.elf}"
DEFAULT_BOOTSTRAP="${FRAME_NIR_BOOTSTRAP:-/home/steamos/hexecutor-hand-viewer/dsp/bootstrap4.elf}"
DEFAULT_CHUNK_PROBE="${FRAME_NIR_CHUNK_PROBE:-/home/steamos/hexecutor-hand-viewer/dsp/chunk_probe.elf}"

use_cdsp=false
filtered_args=()
for arg in "$@"; do
    if [[ "$arg" == "--cdsp" ]]; then
        use_cdsp=true
    else
        filtered_args+=("$arg")
    fi
done

if ! has_arg "--models" "${filtered_args[@]}" && [[ -d "$DEFAULT_MODELS" ]]; then
    extra_args+=(--models "$DEFAULT_MODELS")
fi

if [[ "$use_cdsp" == "true" ]] || has_arg "--qnn16-dir" "${filtered_args[@]}"; then
    if ! has_arg "--qnn16-dir" "${filtered_args[@]}" && [[ -d "$DEFAULT_QNN16_DIR" ]]; then
        extra_args+=(--qnn16-dir "$DEFAULT_QNN16_DIR")
    fi
    if ! has_arg "--loader" "${filtered_args[@]}" && [[ -f "$DEFAULT_LOADER" ]]; then
        extra_args+=(--loader "$DEFAULT_LOADER")
    fi
    if ! has_arg "--engine" "${filtered_args[@]}" && [[ -f "$DEFAULT_ENGINE" ]]; then
        extra_args+=(--engine "$DEFAULT_ENGINE")
    fi
    if ! has_arg "--bootstrap" "${filtered_args[@]}" && [[ -f "$DEFAULT_BOOTSTRAP" ]]; then
        extra_args+=(--bootstrap "$DEFAULT_BOOTSTRAP")
    fi
    if ! has_arg "--chunk-probe" "${filtered_args[@]}" && [[ -f "$DEFAULT_CHUNK_PROBE" ]]; then
        extra_args+=(--chunk-probe "$DEFAULT_CHUNK_PROBE")
    fi
fi

sudo -v
# The reader closes the pipe on normal viewer shutdown; SIGPIPE is expected in
# the capture helper. Preserve genuine helper failures and the viewer status.
set +e
sudo "$capture" --raw | bin/frame_nir_hand_viewer "${extra_args[@]}" "${filtered_args[@]}"
status=("${PIPESTATUS[@]}")
set -e
if (( status[1] != 0 )); then exit "${status[1]}"; fi
if (( status[0] != 0 && status[0] != 141 )); then exit "${status[0]}"; fi
