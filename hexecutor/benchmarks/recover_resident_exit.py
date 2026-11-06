#!/usr/bin/env python3
"""Release a resident session whose workers exited but whose service control
stayed at (2,0,8).

Writes nothing unless the retained state is exactly a completed exit: our
loader and preserving bootstrap installed, exit packet published, all three
EXIT markers at its sequence, VTCM lease restored without error, JTLB clear,
clock/HMX restore markers clear, the lease backup stable with a host
fingerprint equal to the target, and one paused eye client. It then
acknowledges the exit, waits for (1,0,1), clears the callback/loader/payload/
arena to the zero state required before a session, and restarts the eye client.
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

import argparse
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
import numpy as np

FRAME=(PATHS.native)
sys.path.insert(0,str(FRAME))
from hexecutor.offline.frame_transport import CTRL, WORK, DMA_BYTES, load_callback, _process_state, _wait_for_single_eye
from hexecutor.offline.hvx_protocol import ARENA_OFFSET, ARENA_BYTES, CODE_BASE, CODE_BYTES, PAYLOAD_BASE, PAYLOAD_BYTES, STATUS, STATUS_OFFSET
from hexecutor.offline.conv_protocol import EXITED, fence_library


def lease_fingerprint_equal(lease):
    fd=os.open('/dev/cdsp_mem',os.O_RDONLY)
    try:
        with mmap.mmap(fd,128<<20,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ) as m:
            raw=bytes(m[96<<20:104<<20]);again=bytes(m[96<<20:104<<20])
    finally:os.close(fd)
    words=np.frombuffer(raw,'<u4').reshape(-1,32)
    hx=int(np.bitwise_xor.reduce(words.reshape(-1)))
    hs=int((words.sum(axis=1,dtype=np.uint64)*np.arange(1,len(words)+1,dtype=np.uint64)).sum(dtype=np.uint64))&0xffffffff
    return raw==again and (hx,hs)==tuple(lease[11:13])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstrap',type=Path,default=(PATHS.native / 'build/qd_graph_boot-c0-preserve-vtcm-guarded-entry.elf'))
    ap.add_argument('--no-eye-client',action='store_true',
                    help='eye tracking is disabled: require zero eyetracking processes and do not restart one')
    a=ap.parse_args()
    code,_=load_callback((PATHS.native / 'build/qd_loader.elf'))
    payload,_=load_callback(a.bootstrap,PAYLOAD_BASE,PAYLOAD_BYTES)
    fence=fence_library((PATHS.native / 'build'))
    with open('/tmp/depth-model-cdsp.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sock=socket.socket(socket.AF_UNIX);sock.settimeout(2)
        sock.connect(b'\0dsp_service');sock.sendall(struct.pack('<I',4))
        message,ancillary,flags,_=sock.recvmsg(64,socket.CMSG_SPACE(16))
        fds=[]
        for level,kind,raw in ancillary:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                values=array.array('i');values.frombytes(raw[:len(raw)-len(raw)%4]);fds.extend(values)
        if message!=b'ADSP' or len(fds)!=2 or flags&socket.MSG_CTRUNC:
            raise RuntimeError('unexpected DSP handshake')
        mem=mmap.mmap(fds[1],DMA_BYTES,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE,offset=0x80000000)
        eye_pid=None;recovered=False
        try:
            u32=lambda off,n=1:struct.unpack_from(f'<{n}I',mem,off)
            co=CODE_BASE-0x87000000;po=PAYLOAD_BASE-0x87000000;ar=ARENA_OFFSET
            control=u32(CTRL,3);packet=u32(ar,4);lease=u32(ar+3328,14)
            markers=[STATUS.unpack_from(mem,ar+STATUS_OFFSET+i*STATUS.size) for i in range(3)]
            print('before',control,hex(u32(WORK+24)[0]),packet,markers,lease[:3],flush=True)
            checks={
                'control (2,0,8) or (2,0,1)':control in ((2,0,8),(2,0,1)),
                'our callback registered':u32(WORK+24)[0]==CODE_BASE,
                'loader matches':mem[co:co+CODE_BYTES]==code,
                'bootstrap matches':mem[po:po+PAYLOAD_BYTES]==payload,
                'exit packet published':packet[0]==0x32564451 and packet[3]==2,
                'all EXIT markers at exit sequence':all(m[0]==packet[2] and m[3]==EXITED for m in markers),
                'lease restored':lease[0]==0x31545651 and lease[1]==2 and lease[2]==0,
                'JTLB clear':u32(ar+3072)[0]==0,
                'clock/HMX restored':u32(ar+256)[0]!=0x41584931 and u32(ar+512)[0]!=0x484d5831,
                'lease fingerprint':lease_fingerprint_equal(lease),
            }
            eyes=[int(p) for p in subprocess.run(['pgrep','-x','eyetracking'],capture_output=True,text=True).stdout.split()]
            if a.no_eye_client:checks['no eye client (eye tracking disabled)']=len(eyes)==0
            else:checks['one paused eye client']=len(eyes)==1 and Path(f'/proc/{eyes[0]}/stat').read_text().split(') ',1)[1][0]=='T'
            failed=[k for k,v in checks.items() if not v]
            if failed:raise RuntimeError(f'not a completed resident exit; nothing written: {failed}')
            eye_pid=None if a.no_eye_client else eyes[0]
            def settle(target,seconds):
                end=time.monotonic()+seconds
                while time.monotonic()<end and not target(u32(CTRL,3)):time.sleep(0.001)
                return u32(CTRL,3)
            if control==(2,0,8):
                struct.pack_into('<I',mem,CTRL+8,1);struct.pack_into('<I',mem,CTRL+4,0);fence()
                control=settle(lambda c:c in ((1,0,1),(2,0,1)),3)
                print('after acknowledge',control,flush=True)
            if control==(2,0,1):
                # The service still holds a dispatch. The arena carries only the
                # exit packet, which the guarded entry rejects before any mapping
                # or VTCM access, so one more dispatch lets it complete.
                struct.pack_into('<I',mem,CTRL+4,1);struct.pack_into('<I',mem,CTRL+8,4);fence()
                sock.sendall(b'work')
                control=settle(lambda c:c[0] in (1,3),5)
                markers=[STATUS.unpack_from(mem,ar+STATUS_OFFSET+i*STATUS.size) for i in range(3)]
                print('after dispatch',control,markers,flush=True)
                if control[0] not in (1,3) or not all(m[0]==packet[2] and m[3]==EXITED for m in markers):
                    raise RuntimeError('DSP exit did not finish; installed memory retained')
                struct.pack_into('<I',mem,CTRL+8,1);struct.pack_into('<I',mem,CTRL+4,0);fence()
                control=settle(lambda c:c==(1,0,1),3)
                print('after release',control,flush=True)
            if control!=(1,0,1):raise RuntimeError('DSP control did not return idle; installed memory retained')
            after=u32(ar+3328,14)
            if after[:3]!=lease[:3] or u32(ar+3072)[0] or not lease_fingerprint_equal(after):
                raise RuntimeError(f'lease state changed during recovery; installed memory retained: {after[:3]}')
            struct.pack_into('<I',mem,WORK+24,0)
            mem[co:co+CODE_BYTES]=bytes(CODE_BYTES)
            mem[po:po+PAYLOAD_BYTES]=bytes(PAYLOAD_BYTES)
            mem[ar:ar+ARENA_BYTES]=bytes(ARENA_BYTES)
            fence();recovered=True
            print('callback cleared',flush=True)
        finally:
            mem.close()
            for fd in fds:os.close(fd)
            sock.close()
        if recovered and eye_pid is not None:
            # Same restart as FrameTransport: a fresh client re-uploads its weights.
            os.kill(eye_pid,signal.SIGTERM);os.kill(eye_pid,signal.SIGCONT)
            end=time.monotonic()+5
            while time.monotonic()<end and _process_state(eye_pid) not in (None,'Z'):time.sleep(0.05)
            time.sleep(1)
            live=_wait_for_single_eye()
            if live is None:
                subprocess.run(['systemd-run','--user','--collect','--unit',f'depth-model-eye-recovery-{os.getpid()}',
                                '/opt/steamvr/tools/eyetracking/bin/linuxarm64/eyetracking','-b','CDSP','-w',
                                '/opt/steamvr/tools/eyetracking/resources/et_dsp_20250610_03136.weights'],check=True)
                live=_wait_for_single_eye()
            print('eye client',live,flush=True)


if __name__=='__main__':main()
