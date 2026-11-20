#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Extract the exact local QAIRT aligned bbb 3x3 leaf as Hexagon words."""

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
import struct, hashlib
provider=Path('/Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/libQnnHtpV75.so')
raw=provider.read_bytes(); assert raw[:7]==b'\x7fELF\x01\x01\x01'
shoff=struct.unpack_from('<I',raw,32)[0]; stride,count=struct.unpack_from('<HH',raw,46)
sections=[struct.unpack_from('<10I',raw,shoff+i*stride) for i in range(count)]
for sec in sections:
    if sec[1]!=11: continue
    strings=sections[sec[6]]; names=raw[strings[4]:strings[4]+strings[5]]
    for off in range(sec[4],sec[4]+sec[5],sec[9]):
        name,addr,size,info,other,si=struct.unpack_from('<IIIBBH',raw,off)
        name=names[name:names.find(b'\0',name)].decode()
        if name!='hmx_v73_convbbb_stride1_aligned': continue
        src=sections[si]; code=raw[src[4]+addr-src[3]:src[4]+addr-src[3]+size]
        assert len(code)==size and size==824 and size%4==0
        out=(PATHS.native / 'kernels/convolution/qd_qnn_hmx_3x3.S')
        lines=['/* Local QAIRT 2.50 hmx_v73_convbbb_stride1_aligned; simulator-only ABI probe. */','.text','.p2align 6','.globl qd_qnn_hmx_3x3','qd_qnn_hmx_3x3:']
        lines += [f'  .word 0x{x:08x}' for x in struct.unpack('<%dI'%(size//4),code)]
        out.write_text('\n'.join(lines)+'\n')
        print(f'{out}: va={addr:#x} bytes={size} code_sha256={hashlib.sha256(code).hexdigest()} provider_sha256={hashlib.sha256(raw).hexdigest()}')
        break
    else: continue
    break
else: raise SystemExit('symbol missing')
