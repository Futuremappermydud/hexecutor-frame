#!/usr/bin/env python3
"""Analyze how the primary program uses memory: tensors, aliases, value lifetimes, peak use.
A value is one version of memory: what a write produces, joined with every descriptor that reads it."""
import collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')); NP = os.path.join(REPO, 'hexecutor/native/program')
sys.path.insert(0, NP)
import compact_v2 as pp
prog = pp.decode(open(os.path.join(REPO, 'samples/depth-model/deployment/programs/primary-v2.nprg'), 'rb').read())
cmds = prog['commands']; tf = prog['tensor_features']; N = len(cmds)
DT = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'int32': 4, 'uint32': 4, 'float32': 4, 'float16': 2, 'int32_quantized': 4}
G = 128
def direct_bytes(t):
    n = DT.get(t['dtype'], 4)
    for x in t['shape']['max_dimensions']: n *= x
    return n
print('pools MiB', [round(x / 1048576, 3) for x in prog['pool_bytes']])
kind = collections.Counter(); byp = collections.Counter(); szb = collections.Counter()
used = set(x for c in cmds for x in list(c[4]) + list(c[5]))
for i in used:
    st = tf[i].get('storage')
    if st is None: kind['no storage'] += 1
    elif 'blocks' in st:
        kind['blocked'] += 1
        for p, o in st['blocks']: byp[('blocked', p)] += 2048
    else:
        kind['direct'] += 1; byp[('direct', st['direct'][0])] += direct_bytes(tf[i])
print('tensors used', len(used), dict(kind))
print('declared bytes by kind and pool (MiB, counts aliases and reuse)', {k: round(v / 1048576, 2) for k, v in sorted(byp.items(), key=str)})
# how many descriptors touch each pool-1 block, and alignment
bs = collections.Counter(); align = collections.Counter(); dsz = collections.Counter()
for i in used:
    st = tf[i].get('storage')
    if st and 'blocks' in st:
        for p, o in st['blocks']:
            if p == 1: bs[o // 2048] += 1; align[o % 2048] += 1
    elif st and st['direct'][0] == 1:
        align[('direct', st['direct'][1] % 2048 == 0)] += 1; dsz[direct_bytes(tf[i]) // 1024] += 1
print('pool-1 block alignment', dict(align))
print('distinct pool-1 blocks in block tables', len(bs), 'descriptors per block: median', sorted(bs.values())[len(bs) // 2], 'max', max(bs.values()))
print('direct pool-1 tensor sizes KiB (top)', dsz.most_common(12))
# Walk program order at 128-byte granules: versions, readers.
def foot(i):
    st = tf[i].get('storage')
    if st is None: return []
    if 'blocks' in st:
        out = []
        for p, o in st['blocks']: out += [(p, o // G + k) for k in range(2048 // G)]
        return out
    p, o = st['direct']; n = direct_bytes(tf[i])
    return [(p, g) for g in range(o // G, (o + max(n, 1) - 1) // G + 1)]
reads = [[] for _ in range(N)]; writes = [[] for _ in range(N)]
for c_i, c in enumerate(cmds):
    for t in c[4]: reads[c_i] += foot(t)
    for t in c[5]: writes[c_i] += foot(t)
    op, var, v = c[1], c[2], c[6]
    if op == 26:
        off = v[3]; spill = []; mem = []
        for j in range(4, len(v), 2):
            mem += [(v[j] & 0x7ff, g) for g in range((v[j] & ~0x7ff) // G, ((v[j] & ~0x7ff) + v[j + 1] - 1) // G + 1)]
            spill += [(6, g) for g in range(off // G, (off + v[j + 1] - 1) // G + 1)]; off += v[j + 1]
        if var == 8: reads[c_i] += spill; writes[c_i] += mem
        else: reads[c_i] += mem; writes[c_i] += spill
    elif op == 44:
        for j in range(v[1] & 0x7fff):
            reads[c_i] += [(1, v[2 + 2 * j] // G + k) for k in range(16)]; writes[c_i] += [(1, v[3 + 2 * j] // G + k) for k in range(16)]
vers = {}  # granule -> (writer, last reader)
life = []  # (pool, granule, writer, lastread)
for i in range(N):
    for g in reads[i]:
        if g in vers: vers[g][1] = i
    for g in writes[i]:
        if g in vers: life.append((g, *vers[g]))
        vers[g] = [i, i]
for g, v in vers.items(): life.append((g, *v))
p1 = [x for x in life if x[0][0] == 1]
dead = sum(1 for x in p1 if x[1] == x[2])
print('pool-1 granule versions', len(p1), 'never read (dead or partial writes)', dead)
# live pool-1 bytes in program order
delta = [0] * (N + 1)
for g, w, r in p1:
    if r > w: delta[w] += 1; delta[r + 1] -= 1
cur = 0; peak = 0; at = 0; curve = []
for i in range(N):
    cur += delta[i]
    if cur > peak: peak = cur; at = i
    curve.append(cur)
MB = G / 1048576
print('live pool-1 in program order: peak %.2f MiB at cmd %d, mean %.2f MiB' % (peak * MB, at, sum(curve) / N * MB))
for q in range(0, N, N // 12): print('  cmd %4d live %.2f MiB' % (q, curve[q] * MB))
rl = sorted(r - w for g, w, r in p1 if r > w)
print('lifetimes in commands: median', rl[len(rl) // 2], 'p90', rl[len(rl) * 9 // 10], 'max', rl[-1])
json.dump(curve, open('/tmp/planes-host/live_curve.json', 'w'))
print('blocks reused: versions per granule', round(len(p1) / len({x[0] for x in p1}), 2))
