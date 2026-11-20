#!/usr/bin/env bash
# Install/bundle hexecutord and the XRService broker shim library.
#
# Usage:
#   hexecutor/packaging/install_hexecutord.sh [INSTALL_DIR]
#
# Default INSTALL_DIR is /home/steamos/hexecutor-broker (on Frame)
# or ./dist (if running locally).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

DEFAULT_DIR="/home/steamos/hexecutor-broker"
if [[ ! -d "/home/steamos" ]]; then
    DEFAULT_DIR="$REPO_ROOT/dist"
fi
INSTALL_DIR="${1:-$DEFAULT_DIR}"

echo "[hexecutor-install] Target install prefix: $INSTALL_DIR"

mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/lib" "$INSTALL_DIR/include" "$INSTALL_DIR/systemd"

# Copy binaries
if [[ -f "$BUILD_DIR/bin/hexecutord" ]]; then
    cp -f "$BUILD_DIR/bin/hexecutord" "$INSTALL_DIR/bin/"
elif [[ -f "$BUILD_DIR/hexecutord" ]]; then
    cp -f "$BUILD_DIR/hexecutord" "$INSTALL_DIR/bin/"
else
    echo "Warning: hexecutord binary not found in $BUILD_DIR. Run cmake --build build first." >&2
fi

# Copy shim library
SHIM_LIB="$(find "$BUILD_DIR" -name "libxrservice_broker_shim.so" -o -name "libxrservice_broker_shim.dylib" | head -n 1 || true)"
if [[ -n "$SHIM_LIB" && -f "$SHIM_LIB" ]]; then
    cp -f "$SHIM_LIB" "$INSTALL_DIR/lib/"
    echo "[hexecutor-install] Bundled shim: $(basename "$SHIM_LIB")"
else
    echo "Warning: libxrservice_broker_shim not found in $BUILD_DIR. Run cmake --build build first." >&2
fi

# Copy C header for direct patching/linking
cp -f "$SCRIPT_DIR/../include/hexecutor/xrservice_broker_shim.h" "$INSTALL_DIR/include/"

# Copy launcher script
cp -f "$SCRIPT_DIR/run_hexecutord.sh" "$INSTALL_DIR/bin/"
chmod +x "$INSTALL_DIR/bin/run_hexecutord.sh" "$INSTALL_DIR/bin/hexecutord" 2>/dev/null || true

# Generate systemd drop-in override for XRService
cat <<EOF > "$INSTALL_DIR/systemd/10-xrservice-broker.conf"
[Service]
Environment="LD_PRELOAD=$INSTALL_DIR/lib/libxrservice_broker_shim.so"
Environment="HEXECUTORD_SOCKET=${HEXECUTORD_SOCKET:-/tmp/hexecutord.sock}"
EOF

echo "[hexecutor-install] Installation complete in $INSTALL_DIR"
echo ""
echo "To patch XRService via systemd drop-in override on SteamOS:"
echo "  sudo mkdir -p /etc/systemd/system/xrservice.service.d"
echo "  sudo cp \"$INSTALL_DIR/systemd/10-xrservice-broker.conf\" /etc/systemd/system/xrservice.service.d/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl restart xrservice"
echo ""
echo "Or run XRService manually with LD_PRELOAD:"
echo "  LD_PRELOAD=\"$INSTALL_DIR/lib/libxrservice_broker_shim.so\" XRService"
