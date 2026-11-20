#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent integer oracle and pool-byte preservation check for dynamic fixtures."""

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

import hashlib
import json
import struct
from pathlib import Path
HERE = (PATHS.native / 'kernels/adapters')
fixture = ((PATHS.native / 'kernels/adapters/fixture-qdrx-2380.bin')).read_bytes()
output = ((PATHS.native / 'kernels/adapters/output-qdrx-2380.bin')).read_bytes()
h = struct.unpack_from('<32I', fixture)
if h[0] != 0x31585151 or h[16] not in (1, 2):
    raise ValueError('invalid dynamic fixture')
_, _, ih, iw, imh, imw, islice, icount, oh, ow, omh, omw, oslice, ocount, ci, co, wfmt, wb, fb = h[:19]
xpad_h, xpad_w, ypad_h, ypad_w, input_zp = h[19:24]
pos = 128
xb = fixture[pos:pos + icount * 2048]; pos += len(xb)
yb = fixture[pos:pos + ocount * 2048]; pos += len(yb)
wr = fixture[pos:pos + wb]; pos += wb
fir = fixture[pos:pos + fb]
fi = struct.unpack('<%di' % (fb // 4), fir)
got = output[128:]
min_delta, max_delta, different, first = 255, -255, 0, None
shape_ok = output[:128] == fixture[:128] and len(got) == len(yb)
effective_h = min(((oh + 7) // 8) * 8, omh - ypad_h, imh - xpad_h)
effective_w = min(((ow + 7) // 8) * 8, omw - ypad_w, imw - xpad_w)
for y in range(effective_h):
    for x in range(effective_w):
        iy, ix = y + xpad_h, x + xpad_w
        ibase = ((iy // 8) * (imw // 8) + ix // 8) * (ci // 32) + islice
        obase = ((y // 8) * (omw // 8) + x // 8) * (co // 32) + oslice
        for out_ch in range(co):
            acc = fi[(out_ch // 32) * 64 + 32 + out_ch % 32]
            for in_ch in range(ci):
                block = ibase + in_ch // 32
                value = xb[block * 2048 + (iy % 8) * 256 + (ix % 8) * 32 + in_ch % 32]
                if wfmt == 1:
                    wi = ((out_ch // 32) * (ci // 4) + in_ch // 4) * 128 + (out_ch % 32) * 4 + in_ch % 4
                else:
                    wi = (((out_ch // 32) * (ci // 32) + in_ch // 32) * 1024 +
                          ((in_ch // 4) % 8) * 128 + (out_ch % 32) * 4 + in_ch % 4)
                weight = wr[wi]
                acc += (value - input_zp) * (weight if weight < 128 else weight - 256)
            mult_word = fi[(out_ch // 32) * 64 + out_ch % 32]
            mult = mult_word & 0xFFFF
            preshift = 34 - ((mult >> 10) & 31) - 14
            expected = max(0, min(255, (((acc >> preshift) * (1024 + (mult & 1023)) + 8192) >> 14)))
            block = obase + out_ch // 32
            actual = got[block * 2048 + (y % 8) * 256 + (x % 8) * 32 + out_ch % 32]
            delta = actual - expected
            min_delta, max_delta = min(min_delta, delta), max(max_delta, delta)
            different += delta != 0
            if delta and first is None:
                first = {'y': y, 'x': x, 'out': out_ch, 'raw_acc': int(acc),
                         'multiplier_word': int(mult_word), 'multiplier_low16': int(mult),
                         'bias': int(fi[(out_ch // 32) * 64 + 32 + out_ch % 32]),
                         'input_zero_point': input_zp, 'expected': int(expected),
                         'actual': int(actual), 'delta': int(delta)}
active = {((ty * (omw // 8) + tx) * (co // 32) + oslice + cb)
          for ty in range((effective_h + 7) // 8) for tx in range((effective_w + 7) // 8) for cb in range(co // 32)}
outside = [i for i in range(ocount) if i not in active and
           got[i * 2048:(i + 1) * 2048] != yb[i * 2048:(i + 1) * 2048]]
passed = shape_ok and min_delta >= 0 and max_delta <= 1 and not outside
report = {'passed': passed, 'offline_only': True, 'sequence': h[1], 'format': ['','Xpb','Xwb'][wfmt],
          'geometry': [oh, ow, ci, co], 'slices': [islice, oslice],
          'input_padding_origin': [xpad_h, xpad_w], 'output_padding_origin': [ypad_h, ypad_w],
          'input_zero_point': input_zp, 'filter_sha256': hashlib.sha256(wr).hexdigest(),
          'fi_sha256': hashlib.sha256(fir).hexdigest(), 'output_values': effective_h * effective_w * co,
          'native_minus_int64_bound': [min_delta, max_delta], 'different_from_integer': different,
          'first_difference': first, 'untouched_output_table_entries': ocount - len(active),
          'outside_active_entries_changed': outside, 'shape_ok': shape_ok}
report_path = (PATHS.native / f'kernels/adapters/qdrx-{h[1]}-adapter-sim.json')
report_path.write_text(json.dumps(report, indent=2) + '\n')
if not passed:
    print(f'FAIL actual command {h[1]}: {json.dumps(report)}')
    raise SystemExit(1)
print(f"PASS actual command {h[1]} {report['format']}/FI + adapter: {report['output_values']} values, delta [{min_delta},{max_delta}], slice/table preservation")
