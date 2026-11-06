#!/usr/bin/env python3
"""Extract the generic V75 QAIRT bbb stride-1 HMX leaf into a separate symbol."""

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

from pathlib import Path
import hashlib
import struct

provider = Path('/Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/libQnnHtpV75.so')
raw = provider.read_bytes()
assert raw[:7] == b'\x7fELF\x01\x01\x01'
sha = hashlib.sha256(raw).hexdigest()
shoff = struct.unpack_from('<I', raw, 32)[0]
stride, count = struct.unpack_from('<HH', raw, 46)
sections = [struct.unpack_from('<10I', raw, shoff + i * stride) for i in range(count)]
for sec in sections:
    if sec[1] != 11:  # SHT_DYNSYM
        continue
    strings = sections[sec[6]]
    names = raw[strings[4]:strings[4] + strings[5]]
    for off in range(sec[4], sec[4] + sec[5], sec[9]):
        name, addr, size, info, other, si = struct.unpack_from('<IIIBBH', raw, off)
        end = names.find(b'\0', name)
        symbol = names[name:end].decode()
        if symbol != 'hmx_v73_convbbb_stride1':
            continue
        src = sections[si]
        code = raw[src[4] + addr - src[3]:src[4] + addr - src[3] + size]
        assert len(code) == size and size == 996 and size % 4 == 0
        out = (PATHS.native / 'kernels/convolution/qd_qnn_hmx_stride1_generic.S')
        lines = [f'/* QAIRT 2.50 {symbol}; V75 simulator-only exact code. */',
                 '.text', '.p2align 6', '.globl qd_qnn_hmx_stride1_generic',
                 'qd_qnn_hmx_stride1_generic:']
        lines += [f'  .word 0x{x:08x}' for x in struct.unpack('<%dI' % (size // 4), code)]
        out.write_text('\n'.join(lines) + '\n')
        print(f'{out}: symbol={symbol} va={addr:#x} bytes={size} '
              f'code_sha256={hashlib.sha256(code).hexdigest()} provider_sha256={sha}')
        break
    else:
        continue
    break
else:
    raise SystemExit('generic symbol missing')
