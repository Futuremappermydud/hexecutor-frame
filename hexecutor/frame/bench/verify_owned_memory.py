"""Prove host/DSP access to every 4 KB page of a private 128 MB allocation."""
import json
import hashlib
import mmap
import os
import secrets
from pathlib import Path
import struct
import sys
import numpy as np
FRAME=Path(__file__).resolve().parents[1];sys.path.insert(0,str(FRAME))
from frame_transport import FrameTransport
from verify_resident_decoder import snapshot

report={'before':snapshot(),'sessions':[],'payload_sha256':hashlib.sha256((FRAME/'build/qd_owned_mem_probe.elf').read_bytes()).hexdigest()}
report_path=FRAME/'build/owned-memory-nonce-validation.json'
def save():report_path.write_text(json.dumps(report,indent=2)+'\n')
for _ in range(2):
    with FrameTransport(FRAME/'build/qd_loader.elf',FRAME/'build/qd_owned_mem_probe.elf',30,raise_clock=False) as t:
        assert int(Path('/sys/module/cdsp_mem/parameters/size_mb').read_text())==128
        assert int(Path('/sys/module/cdsp_mem/parameters/iova').read_text())==0xa0000000
        fd=os.open('/dev/cdsp_mem',os.O_RDWR)
        try:mem=mmap.mmap(fd,128<<20,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
        finally:os.close(fd)
        try:
            words=np.ndarray((32768,1024),np.uint32,buffer=mem)
            nonce=secrets.randbits(32)
            expected=np.arange(32768,dtype=np.uint32)*4096^np.uint32(nonce)
            words[:,0]=expected;words[:,1]=~expected
            words[:,2]=0xabad1dea;words[:,1023]=expected^np.uint32(0xb519274d)
            def prepare(seq):
                t.arena[:4096]=bytes(4096);struct.pack_into('<I',t.arena,16,nonce);struct.pack_into('<I',t.arena,8,seq);t.fence()
            out=t.invoke(prepare,t.workers_done,lambda:struct.unpack_from('<67I',t.arena,2048))
            report['sessions'].append({'words':out,'nonce':nonce,'pages':out[3],'bytes':128<<20})
            save()
            assert out[0]==0x31544d51 and out[1]==0 and out[3]==32768,out
            assert np.array_equal(words[:,0],expected)
            assert np.array_equal(words[:,1],~expected)
            assert np.array_equal(words[:,1023],expected^np.uint32(0xb519274d))
            assert np.array_equal(words[:,2],~expected)
            assert struct.unpack_from('<I',t.arena,3072)[0]==0
        finally:
            del words,expected
            mem.close()
    report['sessions'][-1]['after']=snapshot()
report['passed']=True
save()
print(json.dumps(report,indent=2))
