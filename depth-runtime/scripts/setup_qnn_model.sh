#!/usr/bin/env bash
# scripts/setup_qnn_model.sh — Automates the QNN DLC conversion using Docker
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QNN_SDK_PATH="${QAIRT_SDK_PATH:-${QNN_SDK_PATH:-${1:-}}}"
IMAGE_NAME="${QNN_CONVERTER_IMAGE:-frame-qnn-converter}"

if [[ -z "$QNN_SDK_PATH" ]]; then
    echo "Usage: $0 /path/to/qairt-sdk" >&2
    echo "       QAIRT_SDK_PATH=/path/to/qairt-sdk $0" >&2
    exit 2
fi

if [[ ! -f "$QNN_SDK_PATH/bin/envsetup.sh" ]]; then
    echo "Error: '$QNN_SDK_PATH' is not a QAIRT SDK directory (bin/envsetup.sh is missing)." >&2
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH." >&2
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "Error: the Docker daemon is not running or is not accessible." >&2
    exit 1
fi

echo "Building QNN conversion image (linux/amd64)..."
DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    --progress=plain \
    -t "$IMAGE_NAME" \
    -f "$PROJECT_DIR/model_research/Dockerfile.qnn" \
    "$PROJECT_DIR/model_research"

echo "Running QNN Conversion..."
docker run --rm \
    --platform linux/amd64 \
    -v "$PROJECT_DIR:/workspace" \
    -v "$QNN_SDK_PATH:/qairt:ro" \
    "$IMAGE_NAME"

echo "Done! The .dlc model has been output to bin/models/hitnet_quantized.dlc"
