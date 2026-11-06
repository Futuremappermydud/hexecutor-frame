#!/usr/bin/env bash
set -euo pipefail
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <OpenXR Vulkan application> [arguments...]" >&2
    exit 2
fi
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XR_API_LAYER_PATH="$project_dir/layers${XR_API_LAYER_PATH:+:$XR_API_LAYER_PATH}"
export XR_ENABLE_API_LAYERS="XR_APILAYER_FRAME_spatial_occlusion${XR_ENABLE_API_LAYERS:+:$XR_ENABLE_API_LAYERS}"
export FRAME_OCCLUSION_STEREO="${FRAME_OCCLUSION_STEREO:-guarded}"
export FRAME_OCCLUSION_SHADER_DIR="$project_dir/shaders"
exec "$@"
