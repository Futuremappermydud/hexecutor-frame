# SPDX-License-Identifier: LGPL-3.0-or-later
"""Read-only check that the CDSP resolves each 2 MB chunk of the private
cdsp_mem window to the same memory the host sees.

After module reloads, stale CDSP translations have resolved IOVA chunks to
other chunks or to foreign memory, so DSP writes there corrupt memory the module
does not own. Run this after every insmod and before any DSP write. The DSP
side (build/qd_chunk_map_probe.elf) only reads; the host writes one tagged word
at the start of each chunk.
"""

from hexecutor.paths import workspace
PATHS = workspace()

import mmap
import os
from pathlib import Path
import secrets
import struct
import numpy as np
from hexecutor.offline.frame_transport import FrameTransport

FRAME=(PATHS.native)
CHUNK=2<<20
import os as _os
WINDOW=int(_os.environ.get('CDSP_WINDOW_MB','128'))<<20  # the probe maps this much (QDO_BYTES)
PROBE=_os.environ.get('CDSP_PROBE_ELF','qd_chunk_map_probe.elf')  # a QDO_SLOTS build for windows above 128 MB

def chunk_map():
    """Return {iova_chunk: resolved_chunk_or_raw_word} for every mismatching chunk."""
    size=int(Path('/sys/module/cdsp_mem/parameters/size_mb').read_text())<<20
    if size<WINDOW:raise RuntimeError(f'cdsp_mem lease is {size>>20} MB; need at least {WINDOW>>20} MB')
    if int(Path('/sys/module/cdsp_mem/parameters/iova').read_text())!=0xa0000000:raise RuntimeError('wrong CDSP IOVA')
    chunks=WINDOW//CHUNK
    with FrameTransport((PATHS.native / 'build/qd_loader.elf'),(PATHS.native / 'build' / PROBE),30,raise_clock=False) as t:
        fd=os.open('/dev/cdsp_mem',os.O_RDWR)
        try:mem=mmap.mmap(fd,WINDOW,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
        finally:os.close(fd)
        try:
            nonce=secrets.randbits(32)&~0x7fe00000  # keep chunk-index bits clean (covers up to 1024 chunks / 1.5-2 GiB)
            words=np.ndarray((chunks,CHUNK//4),np.uint32,buffer=mem)
            words[:,0]=(np.arange(chunks,dtype=np.uint32)<<21)^np.uint32(nonce)
            del words
            def prepare(seq):
                t.arena[:4096]=bytes(4096);struct.pack_into('<I',t.arena,16,nonce)
                struct.pack_into('<I',t.arena,8,seq);t.fence()
            def read():
                head=struct.unpack_from('<8I',t.arena,2048)
                if head[4]==2048:return head+struct.unpack_from(f'<{chunks}I',t.arena,8192)  # table at 8192
                return struct.unpack_from(f'<{100+chunks}I',t.arena,2048)
            out=t.invoke(prepare,t.workers_done,read)
        finally:mem.close()
    if out[0]!=0x31544d51 or out[1] or out[6]:raise RuntimeError(f'chunk-map probe failed: {out[:8]}')
    bad={}
    base=8 if out[4]==2048 else 100
    for ch in range(chunks):
        v=out[base+ch]^nonce
        got=v>>21 if not v&(CHUNK-1) and v>>21<chunks else f'{out[100+ch]:#x}'
        if got!=ch:bad[ch]=got
    return bad

def require_clean_map():
    bad=chunk_map()
    if bad:raise RuntimeError(f'CDSP mis-maps cdsp_mem chunks {bad}; do not write. Reboot, then reload the flushed module.')
