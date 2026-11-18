#!/usr/bin/env bash
# Deploy exported NIR hand tracking models to the headset.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host="${FRAME_HOST:-frame}"
local_models="${1:-$script_dir/../../hand-tracking/.work/nirhands-skeleton-models}"
remote_models="${2:-/home/steamos/nirhands-viewer/models}"

if [[ ! -f "$local_models/manifest.json" ]]; then
    echo "Error: manifest.json not found in $local_models" >&2
    exit 1
fi

echo "Deploying NIR models from: $local_models"
echo "Target host: $host:$remote_models"

ssh "$host" "mkdir -p '$remote_models'"
rsync -avz --progress "$local_models/" "$host:$remote_models/"

echo "Successfully deployed models. Manifest on headset:"
ssh "$host" "cat '$remote_models/manifest.json'"
