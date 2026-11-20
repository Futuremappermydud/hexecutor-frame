#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finish a persistent qwen_dsp session whose host died before sending EXIT.

Requires: dsp_service control (2,0,8), (2,0,1) or (1,1,1), the qd_loader
callback installed, a completed Qwen command (workers idle in their command
loop) or an already-honoured exit packet in the arena, and one paused eye client. Publishes the EXIT packet so thread 0
restores the AXI clock and JTLB, waits for every worker's exit marker, releases
dsp_service, clears loader/payload/arena and restarts the eye client.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import array
import fcntl
import mmap
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import CTRL, WORK, DMA_BYTES, load_callback
from hexecutor.offline.hvx_protocol import ARENA_OFFSET, ARENA_BYTES, CODE_BASE, CODE_BYTES, PAYLOAD_BASE, PAYLOAD_BYTES, STATUS, STATUS_OFFSET
from hexecutor.offline.conv_protocol import EXITED, fence_library, pack_exit

QWEN_MAGIC = 0x334e5751
EXIT_MAGIC = 0x32564451
AXIM_SAVE = 0x7e0000


def wait(cond, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.001)
    return cond()


def main():
    code, _ = load_callback((PATHS.native / 'build/qd_loader.elf'))
    fence = fence_library((PATHS.native / 'build'))
    with open("/tmp/depth-model-cdsp.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sock = socket.socket(socket.AF_UNIX)
        sock.connect(b"\0dsp_service")
        sock.sendall(struct.pack("<I", 4))
        message, ancillary, flags, _ = sock.recvmsg(64, socket.CMSG_SPACE(16))
        fds = []
        for level, kind, raw in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array("i")
                values.frombytes(raw[:len(raw) - len(raw) % 4])
                fds.extend(values)
        if message != b"ADSP" or len(fds) != 2 or flags & socket.MSG_CTRUNC:
            raise RuntimeError("unexpected DSP handshake")
        mem = mmap.mmap(fds[1], DMA_BYTES, flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=0x80000000)
        arena = memoryview(mem)[ARENA_OFFSET:ARENA_OFFSET + ARENA_BYTES]
        get = lambda off: struct.unpack_from("<I", mem, off)[0]
        ctrl = lambda: struct.unpack_from("<3I", mem, CTRL)
        rows = lambda: [STATUS.unpack_from(arena, STATUS_OFFSET + i * STATUS.size) for i in range(3)]
        eye_pid = None
        recovered = False
        try:
            co = CODE_BASE - 0x87000000
            packet = struct.unpack_from("<10I", arena, 0)
            print("before", ctrl(), hex(get(WORK + 24)), [hex(v) for v in packet[:4]], flush=True)
            if ctrl() not in ((1, 1, 1), (2, 0, 8), (2, 0, 1)) or get(WORK + 24) != CODE_BASE:
                raise RuntimeError("state is not an open persistent session")
            if mem[co:co + CODE_BYTES] != code:
                raise RuntimeError("installed loader differs from this checkout")
            if packet[9] != (~packet[2]) & 0xffffffff or packet[0] not in (QWEN_MAGIC, EXIT_MAGIC):
                raise RuntimeError("arena does not hold a Qwen or exit command")
            eyes = [int(p) for p in subprocess.check_output(["pgrep", "-x", "eyetracking"], text=True).split()]
            if len(eyes) != 1 or Path(f"/proc/{eyes[0]}/stat").read_text().split(") ", 1)[1][0] != "T":
                raise RuntimeError("expected one paused eye client")
            eye_pid = eyes[0]
            seq = packet[2]
            exited = lambda s: all(r[0] == s and r[3] == EXITED for r in rows())
            if packet[0] == QWEN_MAGIC:  # workers idle in their command loop
                if not all(r[0] == seq and r[3] != EXITED for r in rows()):
                    raise RuntimeError("workers have not all completed the last command")
                seq += 1
                pack_exit(arena, seq, fence)
                if not wait(lambda: exited(seq), 5):
                    raise RuntimeError(f"workers did not exit: {rows()}; installed memory retained")
            elif not exited(seq):  # exit published earlier (resumed run)
                raise RuntimeError(f"exit packet without exit markers: {rows()}")
            print("workers exited; AXI save marker", hex(get(ARENA_OFFSET + AXIM_SAVE)), flush=True)
            # As in recover_exit.py: acknowledge, and if dsp_service's request
            # thread is not idle, run one more FLOPS wake. The payload sees the
            # exit packet and returns at once.
            if ctrl() == (2, 0, 8):
                struct.pack_into("<I", mem, CTRL + 8, 1)
                fence()
                time.sleep(0.2)
                print("after acknowledge", ctrl(), flush=True)
            if ctrl() != (1, 0, 1):
                if ctrl() not in ((1, 1, 1), (2, 0, 1)):
                    raise RuntimeError(f"unexpected control state {ctrl()}")
                struct.pack_into("<I", mem, CTRL + 4, 1)
                struct.pack_into("<I", mem, CTRL + 8, 4)
                fence()
                sock.sendall(b"work")
                wait(lambda: exited(seq) and ctrl()[0] in (1, 3), 5)
                print("after dispatch", ctrl(), rows(), flush=True)
                if not exited(seq) or ctrl()[0] not in (1, 3):
                    raise RuntimeError("DSP exit did not finish; installed memory retained")
                struct.pack_into("<I", mem, CTRL + 8, 1)
                struct.pack_into("<I", mem, CTRL + 4, 0)
                fence()
            if not wait(lambda: ctrl() == (1, 0, 1), 3):
                raise RuntimeError(f"DSP control did not return idle: {ctrl()}; installed memory retained")
            struct.pack_into("<I", mem, WORK + 24, 0)
            mem[co:co + CODE_BYTES] = bytes(CODE_BYTES)
            po = PAYLOAD_BASE - 0x87000000
            mem[po:po + PAYLOAD_BYTES] = bytes(PAYLOAD_BYTES)
            arena[:] = bytes(ARENA_BYTES)
            fence()
            recovered = True
            print("callback cleared", ctrl(), flush=True)
        finally:
            arena.release()
            mem.close()
            for fd in fds:
                os.close(fd)
            sock.close()
        if recovered:
            os.kill(eye_pid, signal.SIGTERM)
            os.kill(eye_pid, signal.SIGCONT)
            time.sleep(1)
            found = subprocess.run(["pgrep", "-x", "eyetracking"], capture_output=True, text=True)
            live = [int(p) for p in found.stdout.split() if int(p) != eye_pid and
                    Path(f"/proc/{p}/stat").read_text().split(") ", 1)[1][0] != "Z"]
            if not live:
                subprocess.run(["systemd-run", "--user", "--collect", "--unit", f"depth-model-eye-recovery-{os.getpid()}",
                                "/opt/steamvr/tools/eyetracking/bin/linuxarm64/eyetracking",
                                "-b", "CDSP", "-w", "/opt/steamvr/tools/eyetracking/resources/et_dsp_20250610_03136.weights"],
                               check=True)
            print("eye client restarted", flush=True)


if __name__ == "__main__":
    main()
