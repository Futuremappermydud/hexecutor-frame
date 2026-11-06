#!/bin/sh
# Build the CDSP modules against Valve's deckard-linux tree for the running kernel.
# KDIR must be checked out at the running commit: fetch the g<sha> from uname -r
# by full SHA (it may not be on any branch), e.g. in a worktree per kernel version.
set -eu
release=$(uname -r)
base=${release%%-*}
KDIR=${KDIR:-$HOME/work/kernel/deckard-linux-${base%.*}}
MOD=$(cd "$(dirname "$0")/." && pwd)
commit=$(git -C "$KDIR" rev-parse --short=12 HEAD)
[ "${release##*-g}" = "$commit" ] || {
    echo "$KDIR is at $commit, running kernel is $release" >&2
    exit 1
}
# The Frame has no pahole. Without it Kconfig drops DEBUG_INFO_BTF and with it
# SCHED_CLASS_EXT, which moves task_struct fields, including the stack canary
# offset compiled into every module. Modules only need pahole for BTF, which is
# skipped when there is no vmlinux, so report the running kernel's version.
pahole_version=$(zcat /proc/config.gz | sed -n 's/^CONFIG_PAHOLE_VERSION=//p')
PAHOLE=$MOD/.pahole-version-stub
printf '#!/bin/sh\necho v%s.%s\n' "${pahole_version%??}" "${pahole_version#?}" > "$PAHOLE"
chmod +x "$PAHOLE"
if [ ! -f "$KDIR/include/config/kernel.release" ] || [ "$(cat "$KDIR/include/config/kernel.release")" != "$release" ]; then
    zcat /proc/config.gz > "$KDIR/.config"
    sed -i "s/^CONFIG_LOCALVERSION_AUTO=y/# CONFIG_LOCALVERSION_AUTO is not set/; s/^CONFIG_LOCALVERSION=\"\"/CONFIG_LOCALVERSION=\"${release#"$base"}\"/" "$KDIR/.config"
    make -C "$KDIR" -s PAHOLE="$PAHOLE" LOCALVERSION= olddefconfig
    make -C "$KDIR" -s -j8 PAHOLE="$PAHOLE" LOCALVERSION= modules_prepare
fi
test "$(cat "$KDIR/include/config/kernel.release")" = "$release"
zcat /proc/config.gz | grep -v LOCALVERSION | sort > "$MOD/.config.running"
grep -v LOCALVERSION "$KDIR/.config" | sort > "$MOD/.config.build"
diff "$MOD/.config.running" "$MOD/.config.build"
make -C "$KDIR" M="$MOD" PAHOLE="$PAHOLE" KBUILD_MODPOST_WARN=1 LOCALVERSION= modules
