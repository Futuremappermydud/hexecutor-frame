#!/usr/bin/env bash
# curl -fsSL https://<published-pages-url>/get.sh | bash
set -euo pipefail
[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || {
    echo "hexecutord installer requires Linux aarch64 (Steam Frame)" >&2; exit 1;
}
command -v sudo >/dev/null && command -v systemctl >/dev/null && command -v sha256sum >/dev/null || {
    echo "sudo, systemctl and sha256sum are required" >&2; exit 1;
}
kernel=$(uname -r)
base=${HEXECUTOR_RELEASE_BASE:-https://github.com/Futuremappermydud/frame-spatial-runtime/releases/latest/download}
url="$base/hexecutord-linux-aarch64-$kernel.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --retry 3 "$url" -o "$tmp/bundle.tar.gz"
curl -fL --retry 3 "$url.sha256" -o "$tmp/bundle.tar.gz.sha256"
(cd "$tmp" && sha256sum -c bundle.tar.gz.sha256)
mkdir "$tmp/package"
tar -C "$tmp/package" -xzf "$tmp/bundle.tar.gz" --no-same-owner
(cd "$tmp/package" && sha256sum -c SHA256SUMS)
[[ $(cat "$tmp/package/kernel-release") == "$kernel" ]] || {
    echo "bundle kernel does not match $kernel" >&2; exit 1;
}
for file in bin/hexecutord dsp/qd_loader.elf dsp/engine.elf dsp/bootstrap.elf dsp/chunk_probe.elf module/cdsp_mem.ko hexecutord.service COPYING LOADER_LICENSE NOTICE; do
    [[ -f "$tmp/package/$file" ]] || { echo "bundle is missing $file" >&2; exit 1; }
done
sudo bash -s -- "$tmp/package" "$kernel" <<'ROOT'
set -euo pipefail
src=$1
kernel=$2
systemctl stop hexecutord.service 2>/dev/null || true
install -d -m 0755 /opt/hexecutor/bin /opt/hexecutor/dsp "/lib/modules/$kernel/extra"
install -m 0755 "$src/bin/hexecutord" /opt/hexecutor/bin/hexecutord
for name in qd_loader.elf engine.elf bootstrap.elf chunk_probe.elf; do
    install -m 0644 "$src/dsp/$name" "/opt/hexecutor/dsp/$name"
done
install -m 0644 "$src/COPYING" "$src/LOADER_LICENSE" "$src/NOTICE" /opt/hexecutor/
install -m 0644 "$src/module/cdsp_mem.ko" "/lib/modules/$kernel/extra/cdsp_mem.ko"
install -d -m 0755 /opt/hexecutor/module-source
for file in "$src"/module/source/*; do
    install -m 0644 "$file" /opt/hexecutor/module-source/
done
install -m 0644 "$src/hexecutord.service" /etc/systemd/system/hexecutord.service
depmod -a "$kernel"
modprobe cdsp_mem
systemctl daemon-reload
systemctl enable --now hexecutord.service
systemctl is-active --quiet hexecutord.service
ROOT
echo "hexecutord is ready at /run/hexecutor/hexecutord.sock"
