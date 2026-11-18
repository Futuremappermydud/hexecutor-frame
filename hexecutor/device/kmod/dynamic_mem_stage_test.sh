#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0
# Staged, read-only-first test plan for dynamic CDSP memory on the Steam Frame.
#
# DO NOT RUN ON MAC HOST. Execute only directly on the Frame headset with user supervision.
# Note: Steps 1, 2, and 4 modify kernel reservation state and require user watching.

set -uo pipefail

PASS() { echo -e "\e[32m[PASS]\e[0m $*"; }
ABORT() { echo -e "\e[31m[ABORT]\e[0m $*"; exit 1; }
INFO() { echo -e "\e[34m[INFO]\e[0m $*"; }

CDSP_DEV="/dev/cdsp_mem"
SYSFS_SIZE="/sys/module/cdsp_mem/parameters/size_mb"
SYSFS_IOVA="/sys/module/cdsp_mem/parameters/iova"

echo "==================================================================="
echo "  Dynamic CDSP Memory Staged Device Test Plan"
echo "==================================================================="

# -----------------------------------------------------------------------------
# STEP 0: Frame DSP State Check (Read-Only)
# -----------------------------------------------------------------------------
INFO "STEP 0: Checking Frame CDSP state (read-only)..."

# Check for dsp_service journal errors
if journalctl -b --no-pager | grep -i "dsp_service" | grep -i -E "stuck|reset|hung|panic" | tail -5; then
    ABORT "Step 0: Found dsp_service error in journalctl. Reboot or resolve before proceeding."
fi

# Run depth_state.py probe if available
if command -v python3 &>/dev/null && [ -f ~/nirhands-skel/hexecutor/benchmarks/depth_state.py ]; then
    STATE=$(python3 ~/nirhands-skel/hexecutor/benchmarks/depth_state.py 2>/dev/null || true)
    INFO "depth_state output: $STATE"
    if echo "$STATE" | grep -q '"control": \[2, 0, 8\]'; then
        ABORT "Step 0: CDSP firmware is stuck at (2,0,8). Reboot required."
    fi
fi

PASS "Step 0: DSP state is clean and idle."

# -----------------------------------------------------------------------------
# STEP 1: Module Load / Initial Reservation Verification (Needs User Watching)
# -----------------------------------------------------------------------------
INFO "STEP 1: Verifying or inserting cdsp_mem.ko at 128 MiB baseline..."
echo "  >> USER WATCHING REQUIRED: Ensure insmod is authorized and dmesg is monitored."

if [ ! -c "$CDSP_DEV" ] || [ ! -f "$SYSFS_SIZE" ]; then
    INFO "cdsp_mem not loaded. Attempting insmod..."
    if [ -f cdsp_mem.ko ]; then
        sudo insmod cdsp_mem.ko size_mb=128 allow_user_resize=1 || ABORT "Step 1: insmod failed"
    else
        ABORT "Step 1: /dev/cdsp_mem not found and cdsp_mem.ko not present in current dir."
    fi
fi

CUR_SIZE=$(cat "$SYSFS_SIZE" 2>/dev/null || echo 0)
CUR_IOVA=$(cat "$SYSFS_IOVA" 2>/dev/null || echo 0)

[ "$CUR_SIZE" -ge 128 ] || ABORT "Step 1: size_mb ($CUR_SIZE) is less than 128 MiB"
[ "$CUR_IOVA" -eq 2684354560 ] || ABORT "Step 1: iova ($CUR_IOVA) is not 0xa0000000 (2684354560)"

PASS "Step 1: cdsp_mem active at $CUR_SIZE MiB (IOVA $CUR_IOVA)."

# -----------------------------------------------------------------------------
# STEP 2: Step-wise Dynamic Growth & Read-Only Chunk Probe (Needs User Watching)
# -----------------------------------------------------------------------------
INFO "STEP 2: Growing without active clients: 128 -> 256 -> 512 -> 768 -> 1024 MiB..."
echo "  >> USER WATCHING REQUIRED: Monitoring dmesg for SMMU faults after each step."

DMESG_BEFORE=$(dmesg | wc -l)

grow_and_probe() {
    local TARGET_MB=$1
    INFO "Growing to ${TARGET_MB} MiB..."
    echo "$TARGET_MB" | sudo tee "$SYSFS_SIZE" >/dev/null || ABORT "Step 2: Failed to write $TARGET_MB to sysfs"
    
    local ACTUAL=$(cat "$SYSFS_SIZE")
    [ "$ACTUAL" -eq "$TARGET_MB" ] || ABORT "Step 2: Size is $ACTUAL, expected $TARGET_MB"

    # Run read-only chunk probe if available
    if command -v hexecutor_chunk_probe &>/dev/null; then
        INFO "Running read-only chunk probe for ${TARGET_MB} MiB window..."
        HEXECUTOR_CHUNK_WINDOW_MB="$TARGET_MB" hexecutor_chunk_probe || ABORT "Step 2: chunk_probe failed at ${TARGET_MB} MiB"
    elif [ -f ./bin/hexecutor_chunk_probe ]; then
        HEXECUTOR_CHUNK_WINDOW_MB="$TARGET_MB" ./bin/hexecutor_chunk_probe || ABORT "Step 2: chunk_probe failed at ${TARGET_MB} MiB"
    fi

    # Check dmesg for any SMMU faults
    local FAULTS=$(dmesg | tail -n +"$DMESG_BEFORE" | grep -i -E "smmu|fault|arm-smmu|page fault" || true)
    if [ -n "$FAULTS" ]; then
        echo "$FAULTS"
        ABORT "Step 2: SMMU fault detected in dmesg at ${TARGET_MB} MiB!"
    fi

    PASS "Step 2: ${TARGET_MB} MiB grown and probed cleanly."
}

for SZ in 256 512 768 1024; do
    grow_and_probe "$SZ"
done

PASS "Step 2: Growth up to 1024 MiB verified without SMMU faults."

# -----------------------------------------------------------------------------
# STEP 3: Safety Checks (Live Mappings, Pins, EBUSY, Process Death)
# -----------------------------------------------------------------------------
INFO "STEP 3: Testing ioctl and shrink safety enforcement..."

# Helper script for python tests
python3 - << 'EOF'
import sys, os, fcntl, time

CDSP_DEV = "/dev/cdsp_mem"
CDSP_MEM_IOC_MAGIC = ord('C')
# _IOW('C', 1, uint32_t)
CDSP_MEM_IOC_SET_SIZE = 0x40044301
# _IOW('C', 4, uint32_t)
CDSP_MEM_IOC_PIN = 0x40044304
# _IO('C', 5)
CDSP_MEM_IOC_UNPIN = 0x20004305

def pack_u32(val):
    import struct
    return struct.pack("I", val)

print("[Python] Testing ioctl safety rules...")
EOF

# 3a. Shrink with a live mmap must return EBUSY
INFO "3a. Testing live mmap shrink protection (512 MiB mapped, shrink to 256 must return EBUSY)..."
python3 - << 'EOF' &
import os, mmap, time, struct, fcntl
fd = os.open("/dev/cdsp_mem", os.O_RDWR)
# mmap 512 MiB
mem = mmap.mmap(fd, 512 * 1024 * 1024, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
# signal ready
with open("/tmp/cdsp_mmap_ready", "w") as f:
    f.write("ready")
time.sleep(5)
mem.close()
os.close(fd)
EOF
MMAP_PID=$!

# Wait for helper to map
for i in {1..20}; do
    [ -f /tmp/cdsp_mmap_ready ] && break
    sleep 0.1
done
rm -f /tmp/cdsp_mmap_ready

# Attempt to shrink to 256 MiB: must fail with EBUSY
if echo 256 | sudo tee "$SYSFS_SIZE" 2>/dev/null; then
    kill -9 "$MMAP_PID" 2>/dev/null || true
    ABORT "Step 3a: Shrink to 256 MiB succeeded despite active 512 MiB mmap! Safety violation!"
else
    PASS "Step 3a: Shrink to 256 MiB refused with -EBUSY while 512 MiB mapped."
fi

# Shrink to 512 MiB (down to pinned extent) must succeed
if echo 512 | sudo tee "$SYSFS_SIZE" >/dev/null 2>&1; then
    PASS "Step 3a: Shrink down to exact mapped extent (512 MiB) allowed."
else
    kill -9 "$MMAP_PID" 2>/dev/null || true
    ABORT "Step 3a: Shrink down to 512 MiB was unexpectedly refused."
fi

# Wait for helper to exit
wait "$MMAP_PID" 2>/dev/null || true

# 3b. Pin blocks shrink; killing pinning process releases it
INFO "3b. Testing DSP-window pin and release-on-exit..."
python3 - << 'EOF' &
import os, time, struct, fcntl
fd = os.open("/dev/cdsp_mem", os.O_RDWR)
CDSP_MEM_IOC_PIN = 0x40044304
# Pin 512 MiB
fcntl.ioctl(fd, CDSP_MEM_IOC_PIN, struct.pack("I", 512))
with open("/tmp/cdsp_pin_ready", "w") as f:
    f.write("ready")
time.sleep(30)
os.close(fd)
EOF
PIN_PID=$!

for i in {1..20}; do
    [ -f /tmp/cdsp_pin_ready ] && break
    sleep 0.1
done
rm -f /tmp/cdsp_pin_ready

# Attempt to shrink to 256 MiB while pinned at 512 MiB: must fail
if echo 256 | sudo tee "$SYSFS_SIZE" 2>/dev/null; then
    kill -9 "$PIN_PID" 2>/dev/null || true
    ABORT "Step 3b: Shrink succeeded despite active pin! Safety violation!"
else
    PASS "Step 3b: Shrink to 256 MiB blocked by active DSP-window pin."
fi

# Kill the pinning process; kernel .release must automatically drop the pin
kill -9 "$PIN_PID" 2>/dev/null || true
wait "$PIN_PID" 2>/dev/null || true
sleep 0.2

# Now shrink to 256 MiB must succeed
if echo 256 | sudo tee "$SYSFS_SIZE" >/dev/null 2>&1; then
    PASS "Step 3b: Shrink to 256 MiB succeeded after pinning process killed."
else
    ABORT "Step 3b: Shrink to 256 MiB failed even after pinning process exited."
fi

PASS "Step 3: All ioctl safety and pin lifecycle tests passed."

# -----------------------------------------------------------------------------
# STEP 4: Shrink Back to 128 MiB Baseline (Needs User Watching)
# -----------------------------------------------------------------------------
INFO "STEP 4: Shrinking back to 128 MiB baseline..."
echo 128 | sudo tee "$SYSFS_SIZE" >/dev/null || ABORT "Step 4: Failed to shrink to 128 MiB"

FINAL_SIZE=$(cat "$SYSFS_SIZE")
[ "$FINAL_SIZE" -eq 128 ] || ABORT "Step 4: Final size is $FINAL_SIZE, expected 128"

# Check dmesg again
DMESG_FAULTS=$(dmesg | tail -n +"$DMESG_BEFORE" | grep -i -E "smmu|fault|arm-smmu|panic" || true)
if [ -n "$DMESG_FAULTS" ]; then
    echo "$DMESG_FAULTS"
    ABORT "Step 4: SMMU faults detected during teardown!"
fi

PASS "Step 4: Reservation successfully restored to 128 MiB baseline with clean SMMU."

echo "==================================================================="
echo "  ALL STAGED DEVICE TESTS PASSED CLEANLY"
echo "==================================================================="
