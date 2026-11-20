#!/bin/sh
# SPDX-License-Identifier: LGPL-3.0-or-later
set -eu
cd "$(dirname "$0")/."
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"
export DISPLAY=:0
unset WAYLAND_DISPLAY
export GDK_BACKEND=x11
# launch_cpu.sh sets these for the LFM3 CPU panel.
title="${FRAME_CHAT_TITLE:-Frame NPU Chat}"
steam_game="${FRAME_CHAT_STEAM_GAME:-2000600601}"
/usr/bin/python3 frame_chat.py &
app_pid=$!
trap 'kill "$app_pid" 2>/dev/null || true' EXIT INT TERM
attempt=0
while [ "$attempt" -lt 80 ] && kill -0 "$app_pid" 2>/dev/null; do
    window=$(xwininfo -root -children 2>/dev/null | awk -v t="\"$title\"" 'index($0, t) {print $1; exit}')
    if [ -n "$window" ]; then
        xprop -id "$window" -f STEAM_GAME 32c -set STEAM_GAME "$steam_game"
        break
    fi
    attempt=$((attempt + 1))
    sleep 0.25
done
wait "$app_pid"
