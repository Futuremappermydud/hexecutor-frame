#!/bin/sh
# Run on the Frame as root (once). Installs cdsp_mem.ko and a systemd unit that loads it at every boot
# at the 128 MiB baseline with allow_user_resize=1, so unprivileged hexecutord can grow the lease.
# Re-run after rebuilding the module for a new kernel.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
install -D -m 0644 "$here/cdsp_mem.ko" /var/lib/hexecutor/cdsp_mem.ko
install -m 0644 "$here/cdsp-mem.service" /etc/systemd/system/cdsp-mem.service
systemctl daemon-reload
systemctl enable cdsp-mem.service
[ -e /dev/cdsp_mem ] || systemctl start cdsp-mem.service
cat /sys/module/cdsp_mem/parameters/size_mb /sys/module/cdsp_mem/parameters/iova
