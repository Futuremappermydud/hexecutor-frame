#!/usr/bin/env python3
"""Export sequence 247's exact pool1 block offsets for the SDK scatter sim."""
import struct
import sys
from pathlib import Path

FRAME=Path(__file__).resolve().parents[1]
ROOT=FRAME.parent
sys.path.insert(0,str(FRAME))
import resident_image

graph=resident_image.unpack((FRAME/'build/resident-primary.qdrx').read_bytes())
tensors={t['id']:t for t in graph['tensors']}
cmd=next(c for c in graph['commands'] if c['sequence']==247)
x,y=(tensors[cmd['inputs'][0]],tensors[cmd['outputs'][0]])
w,fi=(tensors[cmd['inputs'][1]],tensors[cmd['inputs'][2]])
assert x['storage']['blocks'] and y['storage']['blocks']
assert all(b['pool']==1 for b in x['storage']['blocks']+y['storage']['blocks'])
xo=[b['offset'] for b in x['storage']['blocks']]
yo=[b['offset'] for b in y['storage']['blocks']]
assert w['storage']['direct']['pool']==fi['storage']['direct']['pool']==1
wo=w['storage']['direct']['offset'];fo=fi['storage']['direct']['offset']
def nbytes(t):
    n=t['element_bytes']
    for d in t['shape']['dimensions']:n*=d
    return n
assert wo%128==fo%128==0
path=Path(__file__).with_name('direct-3x3-qdrx247-offsets.bin')
header=struct.pack('<8I',0x324d4353,247,len(xo),len(yo),wo,fo,
                   nbytes(w),nbytes(fi))
path.write_bytes(header+struct.pack('<%dI'%len(xo),*xo)+struct.pack('<%dI'%len(yo),*yo))
print(f'wrote {path}: X={len(xo)} Y={len(yo)}; W/FI offsets {wo:#x}/{fo:#x}')
