#!/bin/sh
# LFM3-1.7B chat panel on the Frame CPU (llama.cpp); see chat_backend_cpu.py.
export FRAME_CHAT_BACKEND=cpu
export FRAME_CHAT_TITLE="Frame LFM3 Chat"
export FRAME_CHAT_STEAM_GAME=2000600602
exec "$(dirname "$0")/launch.sh"
