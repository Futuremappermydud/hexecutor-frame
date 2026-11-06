#!/usr/bin/env python3
"""Independent int64 oracle for the adapter's actual command-1195 fixture."""
import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAGIC = 0x31585151
fixture = (HERE / 'fixture-qdrx-1195.bin').read_bytes()
result = (HERE / 'output-qdrx-1195.bin').read_bytes()
header = struct.unpack_from('<9I', fixture)
magic, seq, h, w, ci, co, ib, wb, fb = header
assert magic == MAGIC and seq == 1195 and (h, w, ci, co, ib, wb, fb) == (8, 64, 128, 128, 32, 16384, 1024)
assert struct.unpack_from('<9I', result) == header
at = 36
x_blocks = fixture[at:at + ib * 2048]
at += ib * 2048
w_raw = fixture[at:at + wb]
at += wb
fi_raw = fixture[at:at + fb]
got_blocks = result[36:]
assert len(got_blocks) == ib * 2048


def get_dense(blocks, y, x, c, channels):
    cb = channels // 32
    index = ((y // 8) * (w // 8) + x // 8) * cb + c // 32
    return blocks[index * 2048 + (y % 8) * 256 + (x % 8) * 32 + c % 32]


def get_weight(i, o):
    off = ((o // 32) * (ci // 4) + i // 4) * 128 + (o % 32) * 4 + i % 4
    v = w_raw[off]
    return v if v < 128 else v - 256


fi = struct.unpack('<%di' % (fb // 4), fi_raw)
got = []
want = []
different = 0
delta_min, delta_max = 255, -255
for y in range(h):
    for x in range(w):
        for o in range(co):
            acc = fi[(o // 32) * 64 + 32 + o % 32]
            for i in range(ci):
                acc += get_dense(x_blocks, y, x, i, ci) * get_weight(i, o)
            mult = fi[(o // 32) * 64 + o % 32] & 0xffff
            shift = 34 - ((mult >> 10) & 31)
            pre = shift - 14
            assert 0 <= pre <= 31
            q = ((((acc >> pre) * (1024 + (mult & 1023))) + 8192) >> 14)
            want.append(max(0, min(255, q)))
            v = get_dense(got_blocks, y, x, o, co)
            got.append(v)
            delta = v - want[-1]
            different += delta != 0
            delta_min = min(delta_min, delta)
            delta_max = max(delta_max, delta)
assert delta_min >= 0 and delta_max <= 1, (delta_min, delta_max)
report = {
    'passed': True,
    'offline_only': True,
    'actual_qdrx_sequence': seq,
    'actual_filter_tensor': 1200,
    'actual_filter_source_tensor': 1199,
    'actual_fi_tensor': 1202,
    'actual_fi_source_tensor': 1201,
    'geometry': {'h': h, 'w': w, 'ci': ci, 'co': co},
    'activation_fixture': 'synthetic deterministic blocked u8 values',
    'filter_sha256': hashlib.sha256(w_raw).hexdigest(),
    'fi_sha256': hashlib.sha256(fi_raw).hexdigest(),
    'outputs': len(got),
    'different_from_int64_oracle': different,
    'native_minus_oracle_bound': [delta_min, delta_max],
    'workspace_used_bytes': 0x56800,
}
(HERE / 'qdrx-1195-adapter-sim.json').write_text(json.dumps(report, indent=2) + '\n')
print(f"PASS actual command {seq} Xpb/Fi + simulator adapter: {len(got)} outputs, delta [{delta_min},{delta_max}]")
