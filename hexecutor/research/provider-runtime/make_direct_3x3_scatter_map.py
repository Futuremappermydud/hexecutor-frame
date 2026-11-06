#!/usr/bin/env python3
"""Export sequence 247's exact pool1 block offsets for the SDK scatter sim."""

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

import struct
import sys
from pathlib import Path

FRAME=(PATHS.native)
ROOT=(PATHS.repo)
sys.path.insert(0,str(FRAME))
import depth_model.lowering.resident_image as resident_image
graph=resident_image.unpack(((PATHS.native / 'build/resident-primary.qdrx')).read_bytes())
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
path=(PATHS.repo / 'hexecutor/research/provider-runtime/direct-3x3-qdrx247-offsets.bin')
header=struct.pack('<8I',0x324d4353,247,len(xo),len(yo),wo,fo,
                   nbytes(w),nbytes(fi))
path.write_bytes(header+struct.pack('<%dI'%len(xo),*xo)+struct.pack('<%dI'%len(yo),*yo))
print(f'wrote {path}: X={len(xo)} Y={len(yo)}; W/FI offsets {wo:#x}/{fo:#x}')
