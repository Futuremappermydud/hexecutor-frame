#!/usr/bin/env bash
# Build the MNN Vulkan runtime once in the Frame user's persistent home.
set -euo pipefail

FRAME_HOST="${FRAME_HOST:-frame}"
MNN_COMMIT="024a946b0b8fcf87c8a418229fadd4cd7858ffba"

ssh "$FRAME_HOST" "MNN_COMMIT='$MNN_COMMIT' bash -s" <<'REMOTE'
set -euo pipefail

prefix="$HOME/.local/frame-mnn"
source_dir="$HOME/.cache/frame-mnn-src"
build_dir="$HOME/.cache/frame-mnn-build"

if [[ -f "$prefix/lib/libMNN.so" && -f "$prefix/include/MNN/Interpreter.hpp" ]]; then
    echo "[Vulkan AI] MNN is already installed at $prefix"
    exit 0
fi

if [[ ! -d "$source_dir/.git" ]]; then
    git clone --depth 1 https://github.com/alibaba/MNN.git "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$MNN_COMMIT"
git -C "$source_dir" checkout -q "$MNN_COMMIT"

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_VULKAN=ON \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_BUILD_BENCHMARK=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_SEP_BUILD=OFF
cmake --build "$build_dir" --target MNN -j6
cmake --install "$build_dir" --prefix "$prefix"

echo "[Vulkan AI] Installed MNN $MNN_COMMIT at $prefix"
REMOTE
