#!/usr/bin/env python3
"""Remove this app's orphaned callback only when the Frame DSP is idle."""

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

sys.path.insert(0,str((PATHS.native)))

from hexecutor.offline.frame_transport import CTRL, WORK, DMA_BYTES, load_callback
from hexecutor.offline.hvx_protocol import ARENA_OFFSET, ARENA_BYTES, CODE_BASE, CODE_BYTES, PAYLOAD_BASE, PAYLOAD_BYTES, STATUS, STATUS_OFFSET
from hexecutor.offline.conv_protocol import EXITED, fence_library


def main():
    frame=(PATHS.native)
    code,_=load_callback((PATHS.native / 'build/qd_loader.elf'))
    payload,_=load_callback((PATHS.native / 'build/hmx_q4_matvec.elf'),PAYLOAD_BASE,PAYLOAD_BYTES)
    with open('/tmp/depth-model-cdsp.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sock=socket.socket(socket.AF_UNIX)
        sock.connect(b'\0dsp_service')
        sock.sendall(struct.pack('<I',4))
        message, ancillary, flags, _=sock.recvmsg(64,socket.CMSG_SPACE(16))
        fds=[]
        for level,kind,raw in ancillary:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                data=array.array('i');data.frombytes(raw[:len(raw)-len(raw)%4]);fds.extend(data)
        if message!=b'ADSP' or len(fds)!=2 or flags&socket.MSG_CTRUNC:
            raise RuntimeError('unexpected DSP handshake')
        eye=[]
        recovered=False
        try:
            mem=mmap.mmap(fds[1],DMA_BYTES,flags=mmap.MAP_SHARED,
                          prot=mmap.PROT_READ|mmap.PROT_WRITE,offset=0x80000000)
            try:
                eye=[int(x) for x in subprocess.check_output(['pgrep','-x','eyetracking'],text=True).split()]
                if len(eye)!=1:raise RuntimeError('expected one eye tracker')
                os.kill(eye[0],signal.SIGSTOP)
                control=struct.unpack_from('<3I',mem,CTRL)
                pointer=struct.unpack_from('<I',mem,WORK+24)[0]
                co=CODE_BASE-0x87000000; po=PAYLOAD_BASE-0x87000000
                markers=[STATUS.unpack_from(mem,ARENA_OFFSET+STATUS_OFFSET+i*STATUS.size) for i in range(3)]
                print('control',control,'pointer',hex(pointer),'markers',markers,flush=True)
                if pointer!=CODE_BASE:
                    raise RuntimeError('callback is not ours')
                if mem[co:co+CODE_BYTES]!=code or mem[po:po+PAYLOAD_BYTES]!=payload:
                    raise RuntimeError('code page differs from this app')
                if not all(mark[3] in (0,EXITED) for mark in markers):
                    raise RuntimeError('worker exit markers are unexpected')
                if control==(2,0,8):
                    seq=struct.unpack_from('<I',mem,ARENA_OFFSET+8)[0]
                    status=struct.unpack_from('<13I',mem,ARENA_OFFSET+2048)
                    if not all(mark[0]==seq and mark[3]==EXITED for mark in markers):
                        raise RuntimeError('completed-job markers do not match')
                    if status[0]!=6 or status[1] or status[2] or status[3]!=32 or status[7] or status[10] or status[11] or status[12]!=68:
                        raise RuntimeError('completed-job HMX status is not clean')
                    struct.pack_into('<I',mem,CTRL+8,1)
                    struct.pack_into('<I',mem,CTRL+4,0)
                    fence_library((PATHS.native / 'build'))()
                    end=time.monotonic()+3
                    while time.monotonic()<end and struct.unpack_from('<3I',mem,CTRL)!=(1,0,1):
                        time.sleep(0.001)
                    control=struct.unpack_from('<3I',mem,CTRL)
                    print('released completed job',control,flush=True)
                if control!=(1,0,1):
                    raise RuntimeError('DSP did not return to idle')
                struct.pack_into('<I',mem,WORK+24,0)
                mem[co:co+CODE_BYTES]=bytes(CODE_BYTES)
                mem[po:po+PAYLOAD_BYTES]=bytes(PAYLOAD_BYTES)
                mem[ARENA_OFFSET:ARENA_OFFSET+ARENA_BYTES]=bytes(ARENA_BYTES)
                fence_library((PATHS.native / 'build'))()
                recovered=True
                print('recovered idle callback',flush=True)
            finally:
                mem.close()
        finally:
            if recovered:
                for pid in eye:
                    os.kill(pid,signal.SIGTERM)
                    os.kill(pid,signal.SIGCONT)
                end=time.monotonic()+3
                while time.monotonic()<end:
                    found=subprocess.run(['pgrep','-x','eyetracking'],capture_output=True,text=True)
                    live=[int(p) for p in found.stdout.split()
                          if int(p) not in eye and Path(f'/proc/{p}/stat').read_text().split(') ',1)[1][0]!='Z']
                    if live:break
                    time.sleep(0.1)
                if not live:
                    unit=f'depth-model-eye-recovery-{os.getpid()}'
                    subprocess.run(['systemd-run','--user','--collect','--unit',unit,
                                    '/opt/steamvr/tools/eyetracking/bin/linuxarm64/eyetracking',
                                    '-b','CDSP','-w',
                                    '/opt/steamvr/tools/eyetracking/resources/et_dsp_20250610_03136.weights'],check=True)
                print('eye client restarted',flush=True)
            for fd in fds:os.close(fd)
            sock.close()


if __name__=='__main__':main()
