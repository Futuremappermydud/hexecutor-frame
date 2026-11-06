#!/bin/sh
# Load the CDSP weight buffer for the whole-token Qwen engine (needs sudo).
# Not persistent: run again after every reboot, before starting the chat.
set -eu
cd "$(dirname "$0")/."
[ -e /dev/cdsp_mem ] && { echo "cdsp_mem already loaded"; exit 0; }
[ -f cdsp_mem.ko ] || ./build.sh
sudo insmod cdsp_mem.ko size_mb=768
ls -l /dev/cdsp_mem
