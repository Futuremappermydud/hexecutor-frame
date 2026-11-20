#!/usr/bin/env bash
# Build a release bundle from explicitly validated Frame artifacts.
set -euo pipefail
if (( $# != 8 )); then
    echo "usage: $0 OUT.tar.gz HEXECUTORD LOADER ENGINE BOOTSTRAP CHUNK_PROBE CDSP_MEM_KO LOADER_LICENSE" >&2
    exit 2
fi
out=$(realpath -m "$1")
shift
for file in "$@"; do
    [[ -f "$file" ]] || { echo "missing artifact: $file" >&2; exit 1; }
done
script_dir=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$script_dir/../.." && pwd)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/bin" "$stage/dsp" "$stage/module"
mkdir -p "$stage/module/source"
install -m 0755 "$1" "$stage/bin/hexecutord"
install -m 0644 "$2" "$stage/dsp/qd_loader.elf"
install -m 0644 "$3" "$stage/dsp/engine.elf"
install -m 0644 "$4" "$stage/dsp/bootstrap.elf"
install -m 0644 "$5" "$stage/dsp/chunk_probe.elf"
install -m 0644 "$6" "$stage/module/cdsp_mem.ko"
for name in Kbuild cdsp_mem.c cdsp_mem_core.c cdsp_mem_core.h; do
    install -m 0644 "$root/hexecutor/device/kmod/$name" "$stage/module/source/$name"
done
install -m 0644 "$7" "$stage/LOADER_LICENSE"
install -m 0644 "$root/hexecutor/COPYING" "$stage/COPYING"
install -m 0644 "$script_dir/hexecutord.service" "$stage/hexecutord.service"
cat > "$stage/NOTICE" <<'EOF'
Hexecutor host and DSP sources: LGPL-3.0-or-later, see the source headers.
cdsp_mem kernel module: GPL-2.0, source included in module/source/.
Repository: https://github.com/Futuremappermydud/frame-spatial-runtime
The verified loader's separate distribution terms are in LOADER_LICENSE.
EOF
kernel=$(modinfo -F vermagic "$stage/module/cdsp_mem.ko" | awk '{print $1}')
[[ -n "$kernel" ]] || { echo "module has no kernel vermagic" >&2; exit 1; }
printf '%s\n' "$kernel" > "$stage/kernel-release"
(cd "$stage" && sha256sum bin/* dsp/* module/cdsp_mem.ko module/source/* COPYING LOADER_LICENSE NOTICE hexecutord.service kernel-release > SHA256SUMS)
mkdir -p "$(dirname "$out")"
tar -C "$stage" -czf "$out" .
(cd "$(dirname "$out")" && sha256sum "$(basename "$out")" > "$(basename "$out").sha256")
echo "Bundle: $out (kernel $kernel)"
