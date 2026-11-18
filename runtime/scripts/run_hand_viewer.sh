#!/usr/bin/env bash
# Frame's compositor must keep the native-size IMX616 buffers producing frames.
# This is a launch prerequisite; the viewer itself uses only OpenXR and Vulkan.
set -euo pipefail
vrcmd=/opt/steamvr/bin/linuxarm64/vrcmd
policy=camera.autoLowerPassthroughResolution
previous=$("$vrcmd" --settings-int "$policy")
previous=${previous##*=}
if [[ ! "$previous" =~ ^[0-9]+$ ]]; then
    echo 'Cannot read compositor camera resolution policy' >&2
    exit 1
fi
restore() { "$vrcmd" --set-settings-int "$policy" "$previous" >/dev/null; }
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$vrcmd" --set-settings-int "$policy" 0 >/dev/null
current=$("$vrcmd" --settings-int "$policy")
if [[ "${current##*=}" != 0 ]]; then
    echo 'Cannot keep compositor camera buffers at native resolution' >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_ROOT="$(cd "$RUNTIME_DIR/.." && pwd)"

# Default paths for deployed hand tracking artifacts on Frame
DEFAULT_MODELS="${FRAME_HAND_MODELS:-}"
if [[ -z "$DEFAULT_MODELS" ]]; then
    if [[ -d "$APP_ROOT/models/hand-tracking" ]]; then
        DEFAULT_MODELS="$APP_ROOT/models/hand-tracking"
    elif [[ -d "$APP_ROOT/hand-tracking/models/deployment" ]]; then
        DEFAULT_MODELS="$APP_ROOT/hand-tracking/models/deployment"
    elif [[ -d /home/steamos/hexecutor-hand-npu/hand-tracking/.work/validation-final4 ]]; then
        DEFAULT_MODELS="/home/steamos/hexecutor-hand-npu/hand-tracking/.work/validation-final4"
    fi
fi

DEFAULT_LOADER="${FRAME_HAND_LOADER:-${FRAME_DSP_LOADER:-/home/steamos/frame-dsp/quest-depth/wave1-20261101/frame/build/qd_loader.elf}}"

DEFAULT_ENGINE="${FRAME_HAND_ENGINE:-}"
if [[ -z "$DEFAULT_ENGINE" ]]; then
    if [[ -f "$APP_ROOT/dsp/engine-planar.elf" ]]; then
        DEFAULT_ENGINE="$APP_ROOT/dsp/engine-planar.elf"
    elif [[ -f /home/steamos/hexecutor-hand-npu/build/tensor/engine-planar.elf ]]; then
        DEFAULT_ENGINE="/home/steamos/hexecutor-hand-npu/build/tensor/engine-planar.elf"
    fi
fi

DEFAULT_BOOTSTRAP="${FRAME_HAND_BOOTSTRAP:-}"
if [[ -z "$DEFAULT_BOOTSTRAP" ]]; then
    if [[ -f "$APP_ROOT/dsp/bootstrap4.elf" ]]; then
        DEFAULT_BOOTSTRAP="$APP_ROOT/dsp/bootstrap4.elf"
    elif [[ -f /home/steamos/hexecutor-hand-npu/build/tensor/bootstrap4.elf ]]; then
        DEFAULT_BOOTSTRAP="/home/steamos/hexecutor-hand-npu/build/tensor/bootstrap4.elf"
    fi
fi

DEFAULT_CHUNK_PROBE="${FRAME_HAND_CHUNK_PROBE:-}"
if [[ -z "$DEFAULT_CHUNK_PROBE" ]]; then
    if [[ -f "$APP_ROOT/dsp/chunk_probe.elf" ]]; then
        DEFAULT_CHUNK_PROBE="$APP_ROOT/dsp/chunk_probe.elf"
    elif [[ -f /home/steamos/hexecutor-hand-npu/build/tensor/chunk_probe.elf ]]; then
        DEFAULT_CHUNK_PROBE="/home/steamos/hexecutor-hand-npu/build/tensor/chunk_probe.elf"
    fi
fi

has_arg() {
    local target="$1"
    shift
    for a in "$@"; do
        if [[ "$a" == "$target" ]]; then return 0; fi
    done
    return 1
}

extra_args=()
if ! has_arg "--models" "$@" && ! has_arg "--mercury-dir" "$@" && [[ -n "$DEFAULT_MODELS" ]]; then
    extra_args+=(--models "$DEFAULT_MODELS")
fi
if ! has_arg "--loader" "$@" && [[ -n "$DEFAULT_LOADER" ]]; then
    extra_args+=(--loader "$DEFAULT_LOADER")
fi
if ! has_arg "--engine" "$@" && [[ -n "$DEFAULT_ENGINE" ]]; then
    extra_args+=(--engine "$DEFAULT_ENGINE")
fi
if ! has_arg "--bootstrap" "$@" && [[ -n "$DEFAULT_BOOTSTRAP" ]]; then
    extra_args+=(--bootstrap "$DEFAULT_BOOTSTRAP")
fi
if ! has_arg "--chunk-probe" "$@" && [[ -n "$DEFAULT_CHUNK_PROBE" ]]; then
    extra_args+=(--chunk-probe "$DEFAULT_CHUNK_PROBE")
fi

cd "$RUNTIME_DIR"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" exec ./bin/frame_hand_viewer "${extra_args[@]}" "$@"
