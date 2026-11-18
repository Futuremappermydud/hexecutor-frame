#!/usr/bin/env bash
# Start the CDSP broker (hexecutord) on Frame with the hand graphs registered as
# "palm" and "landmark", plus any extra --graph NAME=IMAGE,CONSTANTS arguments.
# FRAME_FEATURENN (or FRAME_STEREOFEATURENN), FRAME_GAZENET and FRAME_GAZENET_VECTOR
# instead register the corresponding stock-protocol adapter graphs from HXFG prefixes (without
# .hxfg/.bin). Add hand graphs explicitly if desired; constant capacity is limited.
# Clients: frame_hand_viewer --broker SOCKET, frame_hand_tracker --broker SOCKET.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

first_existing() {
    for p in "$@"; do
        if [[ -e "$p" ]]; then echo "$p"; return 0; fi
    done
    return 1
}
MODELS="${FRAME_HAND_MODELS:-$(first_existing "$APP_ROOT/models/hand-tracking" \
    /home/steamos/hexecutor-hand-viewer/models/hand-tracking || true)}"
DSP_DIR="${FRAME_HAND_DSP:-$(first_existing "$APP_ROOT/dsp" /home/steamos/hexecutor-hand-viewer/dsp || true)}"
LOADER="${FRAME_HAND_LOADER:-${FRAME_DSP_LOADER:-/home/steamos/frame-dsp/quest-depth/wave1-20261101/frame/build/qd_loader.elf}}"
SOCKET="${HEXECUTORD_SOCKET:-${XDG_RUNTIME_DIR:-/tmp}/hexecutord.sock}"
BIN="${HEXECUTORD:-$(first_existing "$APP_ROOT/build/bin/hexecutord" "$APP_ROOT/runtime/bin/hexecutord" || true)}"
FEATURENN="${FRAME_STEREOFEATURENN:-${FRAME_STEREO_FEATURENN:-${FRAME_FEATURENN:-}}}"
GAZENET="${FRAME_GAZENET:-}"
GAZENET_VECTOR="${FRAME_GAZENET_VECTOR:-}"

for v in DSP_DIR BIN; do
    if [[ -z "${!v}" ]]; then echo "cannot find $v; set it explicitly" >&2; exit 1; fi
done
GRAPHS=()
ADAPTER=()
if [[ -n "$FEATURENN" || -n "$GAZENET" || -n "$GAZENET_VECTOR" ]]; then
    if [[ -n "$FEATURENN" ]]; then
        [[ -f "$FEATURENN.hxfg" && -f "$FEATURENN.bin" ]] ||
            { echo "cannot find stereoFeatureNN graph at $FEATURENN" >&2; exit 1; }
        GRAPHS+=(--graph "featurenn=$FEATURENN.hxfg,$FEATURENN.bin")
        ADAPTER+=(--xrservice-featurenn featurenn)
    fi
    if [[ -n "$GAZENET" ]]; then
        [[ -f "$GAZENET.hxfg" && -f "$GAZENET.bin" ]] ||
            { echo "cannot find Gazenet graph at $GAZENET" >&2; exit 1; }
        GRAPHS+=(--graph "gazenet=$GAZENET.hxfg,$GAZENET.bin")
        ADAPTER+=(--xrservice-gazenet gazenet)
    fi
    if [[ -n "$GAZENET_VECTOR" ]]; then
        [[ -f "$GAZENET_VECTOR.hxfg" && -f "$GAZENET_VECTOR.bin" ]] ||
            { echo "cannot find Gazenet vector graph at $GAZENET_VECTOR" >&2; exit 1; }
        GRAPHS+=(--graph "gazenet_vector=$GAZENET_VECTOR.hxfg,$GAZENET_VECTOR.bin")
        ADAPTER+=(--xrservice-gazenet-vector gazenet_vector)
    fi
else
    [[ -n "$MODELS" ]] || { echo "cannot find MODELS; set it explicitly" >&2; exit 1; }
    if ! grep -q '"enableForLogic": *false' /home/steamos/.config/openvr/config/cv/xrservice/XRServiceSettings.json 2>/dev/null; then
        echo "warning: XRService may still use the CDSP (controller FeatureNN); see docs/hexecutor/operations.md" >&2
    fi
    GRAPHS=(--graph "palm=$MODELS/palm.hxfg,$MODELS/palm.bin"
            --graph "landmark=$MODELS/landmark.hxfg,$MODELS/landmark.bin")
fi

exec "$BIN" --socket "$SOCKET" \
    --loader "$LOADER" \
    --engine "$DSP_DIR/engine-planar.elf" \
    --bootstrap "$DSP_DIR/bootstrap4.elf" \
    --chunk-probe "$DSP_DIR/chunk_probe.elf" \
    "${GRAPHS[@]}" \
    "${ADAPTER[@]}" \
    "$@"
