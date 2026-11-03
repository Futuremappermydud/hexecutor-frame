#!/bin/bash
# scripts/deploy_and_run.sh — Sync code to Steam Frame, compile, and run
set -e

FRAME_HOST="frame"
REMOTE_DIR="/home/steamos/frame-spatial-runtime"

if [[ "${FRAME_INSTALL_MNN:-0}" == "1" ]] && \
   ! ssh "$FRAME_HOST" 'test -f "$HOME/.local/frame-mnn/lib/libMNN.so"'; then
    echo "=== 0. Installing one-time Vulkan AI runtime ==="
    ./scripts/setup_vulkan_ai.sh
fi

echo "=== 1. Syncing codebase to Frame ($FRAME_HOST:$REMOTE_DIR) ==="
rsync -avz --delete \
    --exclude '.git' \
    --exclude 'bin/frame_spatial_runtime' \
    --exclude '*.o' \
    --exclude '.DS_Store' \
    ./ $FRAME_HOST:$REMOTE_DIR/

echo "=== 2. Compiling on Frame ==="
ssh $FRAME_HOST "cd $REMOTE_DIR && make clean && make -j4"

echo "=== 3. Launching Real-Time Spatial Mesh Runtime ==="
ssh -t $FRAME_HOST "cd $REMOTE_DIR && ./bin/frame_spatial_runtime $@"
