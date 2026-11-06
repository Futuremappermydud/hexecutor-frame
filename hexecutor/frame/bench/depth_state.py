#!/usr/bin/env python3
"""Read-only DSP state snapshot, including callback-region ownership."""
import array
import hashlib
import json
import mmap
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys


def main():
    sock=socket.socket(socket.AF_UNIX);sock.settimeout(2)
    fds=[]
    try:
        sock.connect(b'\0dsp_service');sock.sendall(struct.pack('<I',4))
        data,anc,flags,_=sock.recvmsg(64,socket.CMSG_SPACE(16))
        for level,kind,raw in anc:
            if level==socket.SOL_SOCKET and kind==socket.SCM_RIGHTS:
                received=array.array('i');received.frombytes(raw[:len(raw)-len(raw)%received.itemsize]);fds.extend(received)
        assert data==b'ADSP' and len(fds)==2 and not flags&socket.MSG_CTRUNC
        with mmap.mmap(fds[1],0x2000000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0x80000000) as mem:
            g=lambda off:struct.unpack_from('<I',mem,off)[0]
            report={'control':[g(0xa91000+4*i) for i in range(3)],'callback':hex(g(0xb12000+24)),
                    'regions':{name:{'zero':not any(mem[off:off+size]),'sha256':hashlib.sha256(mem[off:off+size]).hexdigest()}
                               for name,off,size in [('loader',0x606000,4096),('payload',0x610000,0x8000),('arena',0x1800000,0x800000)]}}
        proc=subprocess.run(['pgrep','-x','eyetracking'],capture_output=True,text=True)
        report['eye_clients']=[{'pid':int(pid),'state':Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]}
                               for pid in proc.stdout.split() if Path(f'/proc/{pid}/stat').exists()]
        text=json.dumps(report,indent=2)+'\n'
        print(text,end='')
        if len(sys.argv)>1:Path(sys.argv[1]).write_text(text)
    finally:
        for fd in fds:os.close(fd)
        sock.close()


if __name__=='__main__':main()
