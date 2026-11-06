#!/bin/sh
# Build cdsp_bw.ko against Valve's deckard-linux tree for the running kernel.
set -eu
KDIR=${KDIR:-$HOME/work/kernel/deckard-linux}
release=$(uname -r)
if [ ! -f "$KDIR/include/config/kernel.release" ] || [ "$(cat "$KDIR/include/config/kernel.release")" != "$release" ]; then
    zcat /proc/config.gz > "$KDIR/.config"
    sed -i "s/^CONFIG_LOCALVERSION_AUTO=y/# CONFIG_LOCALVERSION_AUTO is not set/; s/^CONFIG_LOCALVERSION=\"\"/CONFIG_LOCALVERSION=\"${release#6.18.0}\"/" "$KDIR/.config"
    make -C "$KDIR" -s LOCALVERSION= olddefconfig
    make -C "$KDIR" -s -j8 LOCALVERSION= modules_prepare
fi
grep -q CONFIG_DEBUG_INFO_BTF_MODULES "$KDIR/include/generated/autoconf.h" ||
    printf '#define CONFIG_DEBUG_INFO_BTF_MODULES 1\n#define CONFIG_DEBUG_INFO_BTF 1\n' >> "$KDIR/include/generated/autoconf.h"
test "$(cat "$KDIR/include/config/kernel.release")" = "$release"
make -C "$KDIR" M="$(cd "$(dirname "$0")" && pwd)" KBUILD_MODPOST_WARN=1 LOCALVERSION= modules
