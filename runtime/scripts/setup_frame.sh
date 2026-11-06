#!/bin/bash
# scripts/setup_frame.sh — Configure Steam Frame for DMA-BUF memory access
set -e

echo "[Setup] Configuring kernel ptrace scope on Frame..."
ssh frame "echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0"

echo "[Setup] Ensuring frame-spatial-runtime directory exists on Frame..."
ssh frame "mkdir -p /home/steamos/frame-spatial-runtime"

echo "[Setup] Configuration complete. Ready for deployment."
