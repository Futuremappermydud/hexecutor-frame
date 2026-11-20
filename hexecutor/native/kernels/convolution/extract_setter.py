#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Extract the local QAIRT stride-1 HMX parameter setter for offline simulation."""

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
import hashlib, struct

provider = Path('/Users/owenelliott/Downloads/qairt/2.50.0.260828/lib/hexagon-v75/unsigned/libQnnHtpV75.so')
raw = provider.read_bytes()
assert raw[:7] == b'\x7fELF\x01\x01\x01'
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
        symbol = names[name:names.find(b'\0', name)].decode()
        if symbol != '_Z22set_hmx_params_stride1P10hmx_paramsmmmmmmm':
            continue
        src = sections[si]
        code = raw[src[4] + addr - src[3]:src[4] + addr - src[3] + size]
        assert len(code) == size and addr == 0x2f35c0 and size == 1400
        out = (PATHS.native / 'kernels/convolution/qd_set_hmx_params_stride1.S')
        lines = ['/* Extracted QAIRT 2.50 stride1 setter; simulator-only ABI probe. */',
                 '.text', '.p2align 6', '.globl qd_set_hmx_params_stride1',
                 'qd_set_hmx_params_stride1:']
        lines += [f'  .word 0x{x:08x}' for x in struct.unpack('<%dI' % (size // 4), code)]
        out.write_text('\n'.join(lines) + '\n')
        print(f'{out}: va={addr:#x} bytes={size} code_sha256={hashlib.sha256(code).hexdigest()} provider_sha256={hashlib.sha256(raw).hexdigest()}')
        break
    else:
        continue
    break
else:
    raise SystemExit('setter symbol missing')
