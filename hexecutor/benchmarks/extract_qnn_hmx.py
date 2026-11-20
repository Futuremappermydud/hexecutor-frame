# SPDX-License-Identifier: LGPL-3.0-or-later
"""Extract a bounded SDK assembly leaf for local ISA experiments."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

from pathlib import Path
import struct,hashlib,json
sdk=Path('/Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/libQnnHtpV75.so')
raw=sdk.read_bytes();assert raw[:7]==b'\x7fELF\x01\x01\x01'
shoff=struct.unpack_from('<I',raw,32)[0];stride,count=struct.unpack_from('<HH',raw,46)
sections=[struct.unpack_from('<10I',raw,shoff+i*stride) for i in range(count)]
wanted={'hmx_v73_convbbb1x1_stride1':'qd_qnn_hmx_1x1'};found={};text=[]
for section in sections:
    if section[1]!=11:continue
    strings=sections[section[6]];names=raw[strings[4]:strings[4]+strings[5]]
    for off in range(section[4],section[4]+section[5],section[9]):
        name,addr,size,info,other,si=struct.unpack_from('<IIIBBH',raw,off);name=names[name:names.find(b'\0',name)].decode()
        if name not in wanted:continue
        src=sections[si];code=raw[src[4]+addr-src[3]:src[4]+addr-src[3]+size];assert len(code)==size and size%4==0
        label=wanted[name];found[name]={'address':hex(addr),'bytes':size,'sha256':hashlib.sha256(code).hexdigest()}
        text.extend([f'/* Local QAIRT leaf {name}, original {addr:#x}, {size} bytes.\n * Fast aligned path only; all other branches must be excluded by caller. */','    .text','    .p2align 6',f'    .globl {label}',f'{label}:'])
        text.extend(f'    .word 0x{word:08x}' for word in struct.unpack('<%dI'%(size//4),code))
assert found.keys()==wanted.keys()
frame=(PATHS.native)
((PATHS.native / 'qd_qnn_hmx_1x1.S')).write_text('\n'.join(text)+'\n')
((PATHS.native / 'build/qnn-hmx-leaf.json')).write_text(json.dumps({'provider_sha256':hashlib.sha256(raw).hexdigest(),'symbols':found},indent=2)+'\n')
print(found)
