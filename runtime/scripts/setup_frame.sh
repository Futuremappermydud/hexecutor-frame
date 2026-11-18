#!/bin/bash
# scripts/setup_frame.sh — Configure Steam Frame for DMA-BUF memory access
set -e
if [[ ${1:-} != '' && ${1:-} != '--qnn' ]]; then
    echo 'Usage: scripts/setup_frame.sh [--qnn]' >&2
    exit 2
fi

echo "[Setup] Configuring kernel ptrace scope on Frame..."
ssh frame "echo frame1 | sudo -S sysctl -w kernel.yama.ptrace_scope=0"

echo "[Setup] Ensuring frame-spatial-runtime directory exists on Frame..."
ssh frame "mkdir -p /home/steamos/frame-spatial-runtime"

if [[ ${1:-} == '--qnn' ]]; then
    echo '[Setup] Checking the QNN cDSP boot state...'
    ssh frame 'fault=/home/steamos/frame-spatial-runtime/runtime/experiments/room-dataset/.npu-fault-boot-id; if [ -f "$fault" ] && [ "$(cat "$fault")" = "$(cat /proc/sys/kernel/random/boot_id)" ]; then echo "Prior cDSP cleanup failed in this boot; reboot the Frame first." >&2; exit 1; fi'
    echo '[Setup] Loading the 128 MB QNN memory lease if needed...'
    ssh frame 'cd /home/steamos/frame-dsp/quest-depth/wave1-20261101 && if [ ! -e /dev/cdsp_mem ]; then echo frame1 | sudo -S insmod frame/kmod/cdsp_mem.ko size_mb=128; fi && test "$(cat /sys/module/cdsp_mem/parameters/size_mb)" = 128 && test "$(cat /sys/module/cdsp_mem/parameters/iova)" = 2684354560'
    echo '[Setup] Verifying every cDSP memory chunk before inference...'
    ssh frame 'cd /home/steamos/frame-dsp/quest-depth/wave1-20261101 && PY=$(python3 -c "import numpy" 2>/dev/null && echo python3 || echo /home/steamos/frame-dsp/quest-depth/.venv314/bin/python3) && $PY frame/bench/chunk_map.py'
fi

echo "[Setup] Configuration complete. Ready for deployment."
