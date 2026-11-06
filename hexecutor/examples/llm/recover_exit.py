#!/usr/bin/env python3
"""Finish a published DSP exit packet after an interrupted persistent session."""

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

FRAME=(PATHS.native)
sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import CTRL, WORK, DMA_BYTES, load_callback
from hexecutor.offline.hvx_protocol import ARENA_OFFSET, ARENA_BYTES, CODE_BASE, CODE_BYTES, PAYLOAD_BASE, PAYLOAD_BYTES, STATUS, STATUS_OFFSET
from hexecutor.offline.conv_protocol import EXITED, fence_library


def main():
    code,_=load_callback((PATHS.native / 'build/qd_loader.elf'))
    payload,_=load_callback((PATHS.native / 'build/hmx_q4_matvec.elf'),PAYLOAD_BASE,PAYLOAD_BYTES)
    with open('/tmp/depth-model-cdsp.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sock=socket.socket(socket.AF_UNIX)
        sock.connect(b'\0dsp_service')
        sock.sendall(struct.pack('<I',4))
        message,ancillary,flags,_=sock.recvmsg(64,socket.CMSG_SPACE(16))
        fds=[]
        for level,kind,raw in ancillary:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                values=array.array('i');values.frombytes(raw[:len(raw)-len(raw)%4]);fds.extend(values)
        if message!=b'ADSP' or len(fds)!=2 or flags&socket.MSG_CTRUNC:
            raise RuntimeError('unexpected DSP handshake')
        mem=mmap.mmap(fds[1],DMA_BYTES,flags=mmap.MAP_SHARED,
                      prot=mmap.PROT_READ|mmap.PROT_WRITE,offset=0x80000000)
        eye_pid=None
        recovered=False
        try:
            co=CODE_BASE-0x87000000;po=PAYLOAD_BASE-0x87000000
            control=struct.unpack_from('<3I',mem,CTRL)
            callback=struct.unpack_from('<I',mem,WORK+24)[0]
            packet=struct.unpack_from('<4I',mem,ARENA_OFFSET)
            print('before',control,hex(callback),packet,flush=True)
            if control not in ((1,1,1),(2,0,8)) or callback!=CODE_BASE or packet[0]!=0x32564451 or packet[3]!=2:
                raise RuntimeError('state is not the expected pending exit')
            if mem[co:co+CODE_BYTES]!=code or mem[po:po+PAYLOAD_BYTES]!=payload:
                raise RuntimeError('installed callback differs from this app')
            eyes=[int(p) for p in subprocess.check_output(['pgrep','-x','eyetracking'],text=True).split()]
            if len(eyes)!=1 or Path(f'/proc/{eyes[0]}/stat').read_text().split(') ',1)[1][0]!='T':
                raise RuntimeError('expected one paused eye client')
            eye_pid=eyes[0]
            statuses=[STATUS.unpack_from(mem,ARENA_OFFSET+STATUS_OFFSET+i*STATUS.size) for i in range(3)]
            if control==(2,0,8):
                if not all(row[0]==packet[2] and row[3]==EXITED for row in statuses):
                    raise RuntimeError('worker exit markers are incomplete')
                struct.pack_into('<I',mem,CTRL+8,1)
                fence_library((PATHS.native / 'build'))()
                time.sleep(0.2)
                control=struct.unpack_from('<3I',mem,CTRL)
                print('after acknowledge',control,flush=True)
            if control!=(1,0,1):
                if control not in ((1,1,1),(2,0,1)):
                    raise RuntimeError('unexpected control state after acknowledgement')
                struct.pack_into('<I',mem,CTRL+4,1)
                struct.pack_into('<I',mem,CTRL+8,4)
                fence_library((PATHS.native / 'build'))()
                sock.sendall(b'work')
                end=time.monotonic()+5
                while time.monotonic()<end:
                    statuses=[STATUS.unpack_from(mem,ARENA_OFFSET+STATUS_OFFSET+i*STATUS.size) for i in range(3)]
                    control=struct.unpack_from('<3I',mem,CTRL)
                    if all(row[0]==packet[2] and row[3]==EXITED for row in statuses) and control[0] in (1,3):
                        break
                    time.sleep(0.001)
                print('after dispatch',control,statuses,flush=True)
                if not all(row[0]==packet[2] and row[3]==EXITED for row in statuses) or control[0] not in (1,3):
                    raise RuntimeError('DSP exit did not finish; installed memory retained')
                struct.pack_into('<I',mem,CTRL+8,1)
                struct.pack_into('<I',mem,CTRL+4,0)
                fence_library((PATHS.native / 'build'))()
            end=time.monotonic()+3
            while time.monotonic()<end and struct.unpack_from('<3I',mem,CTRL)!=(1,0,1):
                time.sleep(0.001)
            if struct.unpack_from('<3I',mem,CTRL)!=(1,0,1):
                raise RuntimeError('DSP control did not return idle; installed memory retained')
            struct.pack_into('<I',mem,WORK+24,0)
            mem[co:co+CODE_BYTES]=bytes(CODE_BYTES)
            mem[po:po+PAYLOAD_BYTES]=bytes(PAYLOAD_BYTES)
            mem[ARENA_OFFSET:ARENA_OFFSET+ARENA_BYTES]=bytes(ARENA_BYTES)
            fence_library((PATHS.native / 'build'))()
            recovered=True
            print('callback cleared',flush=True)
        finally:
            mem.close()
            for fd in fds:os.close(fd)
            sock.close()
        if recovered:
            os.kill(eye_pid,signal.SIGTERM)
            os.kill(eye_pid,signal.SIGCONT)
            time.sleep(1)
            found=subprocess.run(['pgrep','-x','eyetracking'],capture_output=True,text=True)
            live=[int(p) for p in found.stdout.split() if int(p)!=eye_pid and
                  Path(f'/proc/{p}/stat').read_text().split(') ',1)[1][0]!='Z']
            if not live:
                subprocess.run(['systemd-run','--user','--collect','--unit',f'depth-model-eye-recovery-{os.getpid()}',
                                '/opt/steamvr/tools/eyetracking/bin/linuxarm64/eyetracking',
                                '-b','CDSP','-w','/opt/steamvr/tools/eyetracking/resources/et_dsp_20250610_03136.weights'],check=True)
            print('eye client restarted',flush=True)


if __name__=='__main__':main()
