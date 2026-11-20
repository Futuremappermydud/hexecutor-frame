#!/usr/bin/env python3
"""Sensitivity probe: patch per-command delays into a -DQDS_LANE_DELAY engine, no rebuild.

Slowing one class of commands by a known amount and measuring total time shows how much that
class limits the run. The result is a slope: ms of total time per ms of added work.

usage: lane_delay.py ENGINE.elf OUT.elf COSTS.json FRACTION SELECTOR
SELECTOR: comma list of op/var, lane:N, or all. Costs are in ms; the core runs near 1843 MHz.
Prints the total delay added, so the slope can be computed from the measured change.
"""
import json, re, struct, sys
ROOT = __file__.rsplit('/samples/depth-model/', 1)[0]
MAGIC = struct.pack('<4H', 0x4c44, 0x4459, 0x5441, 0x424c)
HZ = 1843.2e6
sys.path[:0] = [ROOT + '/hexecutor/native/program', ROOT + '/hexecutor/program', ROOT + '/samples/depth-model/python/depth_model/lowering', ROOT + '/hexecutor']
import compact_v2 as pp

def lanes():
    h = open(ROOT + '/samples/depth-model/deployment/generated/qnn_lanes_primary.generated.h').read(); out = {}
    for L in range(3):
        for m in re.finditer(r'\{(\d+),', re.search(r'nprg2_qnn_lane%d\[[^\]]*\] = \{(.*?)\n\};' % L, h, re.S).group(1)):
            out[int(m.group(1))] = L
    return out

src, dst, costs, frac, sel = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
cmds = pp.decode(open(ROOT + '/samples/depth-model/deployment/programs/primary-v2.nprg', 'rb').read())['commands']
cost = json.load(open(costs)); lane = lanes(); assert len(cost) == len(cmds)
ops = set(); lns = set(); every = False
for x in sel.split(','):
    if x == 'all': every = True
    elif x.startswith('lane:'): lns.add(int(x[5:]))
    elif x != 'none': ops.add(tuple(map(int, x.split('/'))))
blob = bytearray(open(src, 'rb').read())
at = blob.find(MAGIC); assert at >= 0 and blob.find(MAGIC, at + 1) < 0, 'expected exactly one delay table'
base = at + 8; total = picked = 0
for i, c in enumerate(cmds):
    assert struct.unpack_from('<H', blob, base + 2 * i)[0] == 0, 'engine already patched'
    if i in lane and (every or (c[1], c[2]) in ops or lane[i] in lns):
        d = min(round(cost[i] * 1e-3 * frac * HZ), 65535)
        struct.pack_into('<H', blob, base + 2 * i, d); total += d; picked += 1
open(dst, 'wb').write(blob)
print(json.dumps({'selector': sel, 'fraction': frac, 'commands': picked, 'added_ms': round(total / HZ * 1e3, 4)}))
