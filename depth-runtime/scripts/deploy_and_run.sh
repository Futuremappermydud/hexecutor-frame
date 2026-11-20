#!/bin/bash
# Deploy the monorepo source layout and build the native application/worker.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FRAME_HOST=${FRAME_HOST:-frame}
REMOTE_DIR=${FRAME_REMOTE_DIR:-/home/steamos/frame-spatial-runtime}
if [[ ${FRAME_INSTALL_MNN:-0} == 1 ]] && ! ssh "$FRAME_HOST" 'test -f "$HOME/.local/frame-mnn/lib/libMNN.so"'; then
    "$PROJECT_ROOT/depth-runtime/scripts/setup_vulkan_ai.sh"
fi
rsync -avz --delete --exclude '.git' --exclude '.DS_Store' \
    --exclude '/runtime/' --exclude '/depth-model/' --exclude '/hand-tracking/' \
    --filter=':- .gitignore' "$PROJECT_ROOT/" "$FRAME_HOST:$REMOTE_DIR/"
printf -v remote_root '%q' "$REMOTE_DIR"
printf -v remote_app '%q' "$REMOTE_DIR/depth-runtime"
printf -v launch '%q ' ./bin/frame_spatial_runtime "$@"
ssh "$FRAME_HOST" "cd $remote_root && make build-runtime"
ssh -t "$FRAME_HOST" "cd $remote_app && $launch"
