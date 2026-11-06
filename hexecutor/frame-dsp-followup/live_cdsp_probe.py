#!/usr/bin/env python3
"""Run stock FLOPS, optionally a new worker callback, without restarting DSPs.

Requires the stock Oct 6 firmware. Eye tracker is stopped briefly because the
built-in benchmark reuses its tensor/weights buffers; restarting reuploads them.
No root, device-tree changes, service stops, or firmware-file changes.
"""
import argparse
import array
import hashlib
import mmap
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time

OUT = Path.home() / 'frame-dsp/followup'
BIN = Path('/opt/steamvr/tools/eyetracking/bin/linuxarm64/eyetracking')
WEIGHTS = Path('/opt/steamvr/tools/eyetracking/resources/et_dsp_20250610_03136.weights')
CTRL = 0xa91000
WORK = 0xb12000
CODE = 0x600000

def eye_pids():
    r = subprocess.run(['pgrep', '-x', 'eyetracking'], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]

def waitfor(fn, seconds=2):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if fn():
            return
        time.sleep(.0001)
    raise TimeoutError('DSP protocol timed out')

ap = argparse.ArgumentParser()
ap.add_argument('--callback', type=Path)
ap.add_argument('--vector', action='store_true', help='Initialize and verify 32 int32 HVX lanes per worker')
ap.add_argument('--matmul', action='store_true', help='Initialize and verify 32x32 signed integer HVX matrix multiplication')
args = ap.parse_args()
s = None
mem = None
fds = []
saved_callback = None
saved_code = None
saved_matrix = None
saved_work = None
saved_scratch = None
try:
    firmware = Path('/opt/steamvr/drivers/cv/bin/linuxarm64/cdsp_sm8650.mbn')
    digest = hashlib.sha256(firmware.read_bytes()).hexdigest()
    if digest != '4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c':
        raise RuntimeError('Unrecognized firmware image; fixed offsets are unsafe')
    for pid in eye_pids():
        os.kill(pid, signal.SIGTERM)
    waitfor(lambda: not eye_pids(), 5)
    time.sleep(.2)
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(2)
    s.connect(b'\0dsp_service')
    s.sendall(struct.pack('<I', 4))
    payload, ancillary, _, _ = s.recvmsg(64, socket.CMSG_SPACE(16))
    for level, typ, raw in ancillary:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds.extend(array.array('i', raw))
    if payload != b'ADSP' or len(fds) != 2 or os.fstat(fds[1]).st_size != 0x2000000:
        raise RuntimeError('Unexpected service handshake')
    mem = mmap.mmap(fds[1], 0x2000000, flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=0x80000000)
    def get(off): return struct.unpack_from('<I', mem, off)[0]
    def put(off, value): struct.pack_into('<I', mem, off, value)
    if struct.unpack_from('<QQ', mem, 0xa90000) != (0x1614df651d03f542, 9):
        raise RuntimeError('Unknown CDSP firmware/protocol version')
    print('initial control', [get(CTRL+i*4) for i in range(8)], flush=True)
    if get(CTRL) != 1 or get(CTRL+4) != 0 or get(CTRL+8) != 1:
        raise RuntimeError('CDSP operation interface is not idle')
    saved_callback = get(WORK+24)
    if saved_callback:
        raise RuntimeError('Another worker callback is already installed')
    saved_work = mem[WORK:WORK+32]
    saved_scratch = mem[WORK+0x100:WORK+0xb00]
    if args.callback:
        elf = args.callback.read_bytes()
        entry = struct.unpack_from('<I', elf, 24)[0]
        if entry < 0x87600000 or entry >= 0x87610000 or entry % 4096:
            raise RuntimeError('Callback entry must select a fresh aligned page in the scratch range')
        CODE = entry - 0x87000000
        phoff = struct.unpack_from('<I', elf, 28)[0]
        count = struct.unpack_from('<H', elf, 44)[0]
        chunks = []
        for i in range(count):
            typ, off, va, pa, fs, ms, fl, al = struct.unpack_from('<8I', elf, phoff+i*32)
            if typ == 1 and fs and va >= entry:
                if va + ms > entry+4096:
                    raise RuntimeError('Callback exceeds dedicated 4 KB page')
                chunks.append((va-0x87000000, elf[off:off+fs]))
        if not chunks:
            raise RuntimeError('No callback code found')
        saved_code = mem[CODE:CODE+4096]
        if any(saved_code):
            raise RuntimeError('Chosen code page is occupied')
        for off, raw in chunks:
            mem[off:off+len(raw)] = raw
        mem[WORK+0x100:WORK+0x140] = bytes(64)
        put(WORK+24, entry)
        if args.vector:
            struct.pack_into('<32i', mem, WORK+0x400, *[i*17-200 for i in range(32)])
            struct.pack_into('<32i', mem, WORK+0x500, *[1000-i*3 for i in range(32)])
            mem[WORK+0x800:WORK+0xb00] = bytes(0x300)
        if args.matmul:
            saved_matrix = mem[0x608000:0x60b000]
            if any(saved_matrix):
                raise RuntimeError('Matrix scratch pages are occupied')
            matrix_a = [(r*3+k*5)%7-3 for r in range(32) for k in range(32)]
            matrix_b = [(k*7+c*11)%13-6 for k in range(32) for c in range(32)]
            struct.pack_into('<1024i', mem, 0x608000, *matrix_a)
            struct.pack_into('<1024i', mem, 0x609000, *matrix_b)
            mem[0x60a000:0x60b000] = bytes(4096)
    # Set the request before waking the operations thread. Hold CPU state=1
    # until DSP acknowledges completion, then return CPU state to zero.
    put(CTRL+4, 1)
    put(CTRL+8, 4)
    start = time.monotonic()
    s.sendall(b'work')
    waitfor(lambda: get(CTRL) == 3)
    print('FLOPS completed in %.3f ms' % ((time.monotonic()-start)*1000), flush=True)
    print('benchmark records', struct.unpack_from('<12I', mem, 0xad1084), flush=True)
    if args.callback:
        waitfor(lambda: all(get(WORK+0x108+i*16) == 1 for i in range(3)))
        records = [struct.unpack_from('<4I', mem, WORK+0x100+i*16) for i in range(3)]
        print('new callback records', records, flush=True)
        assert all(r[:3] == (0xc0de0000+i, 63+i, 1) for i,r in enumerate(records))
        if args.vector:
            expected = tuple(800+i*14 for i in range(32))
            for tid in range(3):
                actual = struct.unpack_from('<32i', mem, WORK+0x800+tid*0x100)
                print('HVX worker',tid,'32 lanes',actual,'correct:',actual == expected,flush=True)
                assert actual == expected
        if args.matmul:
            actual = struct.unpack_from('<1024i', mem, 0x60a000)
            expected = tuple(sum(matrix_a[r*32+k]*matrix_b[k*32+c] for k in range(32))
                             for r in range(32) for c in range(32))
            mismatches = [(i,x,y) for i,(x,y) in enumerate(zip(actual,expected)) if x != y]
            print('HVX 32x32 matmul:',len(actual),'outputs,',len(mismatches),'mismatches',
                  'first row:',actual[:32],flush=True)
            assert not mismatches, mismatches[:10]
    put(CTRL+8, 1)
    put(CTRL+4, 0)
    waitfor(lambda: get(CTRL) == 1)
    print('CDSP idle again', flush=True)
finally:
    if mem is not None:
        # Disable future callback calls before restoring the page. Successful
        # requests have already returned all three workers to their wait loop.
        if saved_callback == 0:
            struct.pack_into('<I', mem, WORK+24, 0)
        if get(CTRL) == 3:
            put(CTRL+8, 1)
            put(CTRL+4, 0)
            try:
                waitfor(lambda: get(CTRL) == 1)
            except TimeoutError:
                print('CDSP did not return to idle during cleanup', flush=True)
        if saved_code is not None and get(CTRL) == 1:
            mem[CODE:CODE+4096] = saved_code
        if saved_matrix is not None and get(CTRL) == 1:
            mem[0x608000:0x60b000] = saved_matrix
        if saved_work is not None and get(CTRL) == 1:
            mem[WORK:WORK+32] = saved_work
        if saved_scratch is not None and get(CTRL) == 1:
            mem[WORK+0x100:WORK+0xb00] = saved_scratch
        mem.close()
    if s:
        s.close()
    for fd in fds:
        os.close(fd)
    if not eye_pids():
        with open(OUT/'eye-restored.log','a') as log:
            proc = subprocess.Popen(['taskset','-c','0,1',str(BIN),'-b','CDSP','-w',str(WEIGHTS)],
                                    cwd=BIN.parent,stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True)
        time.sleep(2)
        print('Restored CDSP eye tracking',proc.pid,'running:',proc.poll() is None,flush=True)
