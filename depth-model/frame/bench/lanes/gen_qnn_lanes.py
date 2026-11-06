#!/usr/bin/env python3
"""Generate a main + N worker-lane schedule for the primary model from QNN's run lists.

Dependencies: every conservative memory conflict (2 KiB block footprints of each command's
declared tensors) keeps the order QNN's lists guarantee for it. Conflicts QNN leaves
unordered (always main against worker) stay unordered unless --strict: they race in QNN
and in the two-list engine too. The main list keeps QNN's exact order; worker commands are
list-scheduled by longest remaining path using per-handler costs from a report.

usage: gen_qnn_lanes.py REPORT.json OUT.h [--workers 2] [--events DIR]
"""
import argparse, heapq, json, os, random, re, struct, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
NP = os.path.join(HERE, '..', '..', 'native_program')
sys.path.insert(0, NP)
import compact_v2 as pp

ap = argparse.ArgumentParser()
ap.add_argument('report'); ap.add_argument('out')
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--events')
ap.add_argument('--costs', help='optional JSON list of per-command ms, overrides handler averages')
ap.add_argument('--drop', default='', help='op/var list of no-effect commands to leave out, e.g. 1/10')
ap.add_argument('--move', default='', help='op/var list of main commands (no DMA event) allowed on worker lanes, e.g. 26/8,3/1')
ap.add_argument('--free-main', action='store_true', help='main keeps only DMA queue order; convs may reorder by priority')
ap.add_argument('--model-split', type=float, default=0, help='model only: merge share of a u8 conv becomes a separate task; value = single-thread cost multiple')
ap.add_argument('--scale', default='', help='model only: op/var:factor list applied to costs, e.g. 8/70:0.5,10/13:0')
ap.add_argument('--main-steal-us', type=float, default=0, help='main runs ready worker commands up to this cost when it has no conv ready')
ap.add_argument('--search', type=int, default=0, help='randomized priority search: this many trials, keep the best modeled end')
ap.add_argument('--cpu-weights', default='', help='op/var list of weight copies built with WEIGHTS_CPU_BELOW: they never use the DMA queue')
ap.add_argument('--serial', action='store_true', help='profiling only: every command waits for the previous, so each runs alone with all helpers free')
ap.add_argument('--no-dma-edges', action='store_true', help='model only: leave out the DMA landing edges')
ap.add_argument('--strict', action='store_true', help='also enforce conflicts QNN lists leave unordered (they race in QNN too)')
a = ap.parse_args()

prog = pp.decode(open(os.path.join(NP, 'examples/primary-v2.nprg'), 'rb').read())
cmds = prog['commands']; N = len(cmds)
h = open(os.path.join(NP, 'qnn_runlist_primary.generated.h')).read()
def body(name): return re.search(r'%s\[[^\]]*\]\s*=\s*\{(.*?)\};' % name, h, re.S).group(1)
def entries(name): return [tuple(int(x, 0) for x in e.split(',')) for e in re.findall(r'\{([^{}]*)\}', body(name))]
main, work = entries('nprg2_qnn_primary_main'), entries('nprg2_qnn_primary_worker')
assert len(main) + len(work) == N and all(e[4] == 0 for e in work)
lane = {}; qpos = {}
for L, lst in ((0, main), (1, work)):
    for k, e in enumerate(lst): lane[e[0]] = L; qpos[e[0]] = (L, k)
assert len(lane) == N

# --- costs
r = [x for x in json.load(open(a.report))['runs'] if x['case'] == 'smoke'][-1]
avg = {}
for x in r['handler_profile']:
    mc = x['count'] & 0xffffffff; wc = x['count'] >> 32
    if mc: avg[(0, x['opcode'], x['variant'])] = x['run_ms'] / mc
    if wc: avg[(1, x['opcode'], x['variant'])] = x['validate_ms'] / wc
OVH = 0.00009
cost = [avg.get((lane[i], cmds[i][1], cmds[i][2]), 0.0) + OVH for i in range(N)]
if a.costs:
    per = json.load(open(a.costs)); assert len(per) == N
    cost = [max(float(x), 0.0) + OVH for x in per]

if a.scale:
    sc = {tuple(map(int, x.split(':')[0].split('/'))): float(x.split(':')[1]) for x in a.scale.split(',')}
    cost = [max(cost[i] * sc.get((cmds[i][1], cmds[i][2]), 1.0), 0.00002) for i in range(N)]
# --- memory footprints, GRAN-byte granules. Direct tensors are exact byte ranges;
# blocked tensors cover their whole 2 KiB tiles. Spill, Fill and Zap declare no tensors:
# their memory is in the vector words.
GRAN = 128
DT = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'int32': 4, 'uint32': 4, 'float32': 4, 'float16': 2, 'int32_quantized': 4}
tf = prog['tensor_features']
def span(pool, off, n): return {(pool, g) for g in range(off // GRAN, (off + max(n, 1) - 1) // GRAN + 1)}
def foot(ti):
    t = tf[ti]; st = t.get('storage')
    if st is None: return {('T', ti)}
    if 'blocks' in st:
        r = set()
        for p, o in st['blocks']: r |= span(p, o, 2048)
        return r
    p, o = st['direct']; n = DT.get(t['dtype'], 4)
    for x in t['shape']['max_dimensions']: n *= x
    return span(p, o, n)
R = [set().union(*[foot(i) for i in c[4]]) if c[4] else set() for c in cmds]
W = [set().union(*[foot(i) for i in c[5]]) if c[5] else set() for c in cmds]
for i, c in enumerate(cmds):
    op, var, v = c[1], c[2], c[6]
    if op == 26:
        off = v[3]; spill = set(); mem = set()
        for j in range(4, len(v), 2):
            mem |= span(v[j] & 0x7ff, v[j] & ~0x7ff, v[j + 1]); spill |= span(6, off, v[j + 1]); off += v[j + 1]
        assert var in (8, 9)
        (R[i], W[i]) = (spill, mem) if var == 8 else (mem, spill)
    elif op == 44:
        for j in range(v[1] & 0x7fff):
            R[i] |= span(1, v[2 + 2 * j], 2048); W[i] |= span(1, v[3 + 2 * j], 2048)
    else:
        assert not v or v == (), (op, var)
# Every conflicting pair in program order (not only the last writer), so an edge
# QNN reverses cannot hide another writer of the same granule.
preds = [set() for _ in range(N)]; writers = {}; readers = {}
for i in range(N):
    for g in R[i]: preds[i].update(writers.get(g, ()))
    for g in W[i]:
        preds[i].update(writers.get(g, ())); preds[i].update(readers.get(g, ()))
    for g in R[i]: readers.setdefault(g, set()).add(i)
    for g in W[i]: writers.setdefault(g, set()).add(i)
    preds[i].discard(i)
print('conflict pairs', sum(len(p) for p in preds))

# --- QNN happens-before (vector clocks over the two lists and their checkpoints)
cpprod = {}
for L, lst in ((0, main), (1, work)):
    for k, e in enumerate(lst):
        if e[2]: cpprod.setdefault(e[2], []).append((L, k))
lists = [main, work]; vc = [[None] * len(main), [None] * len(work)]; after = [[None] * len(main), [None] * len(work)]; ptr = [0, 0]
while ptr[0] < len(main) or ptr[1] < len(work):
    moved = False
    for L in (0, 1):
        while ptr[L] < len(lists[L]):
            k = ptr[L]; e = lists[L][k]; base = list(after[L][k - 1]) if k else [0, 0]; ok = True
            if e[1]:
                for pl, pk in cpprod.get(e[1], []):
                    if after[pl][pk] is None: ok = False; break
                    base = [max(base[j], after[pl][pk][j]) for j in (0, 1)]
            if not ok: break
            vc[L][k] = base; x = list(base); x[L] = k + 1; after[L][k] = x; ptr[L] += 1; moved = True
    assert moved, 'QNN lists deadlock'
def hb(p, i):  # QNN guarantees p completes before i starts
    pl, pk = qpos[p]; L, k = qpos[i]
    return vc[L][k][pl] >= pk + 1

# --- lane assignment
def oplist(t): return {tuple(map(int, x.split('/'))) for x in t.split(',') if x}
drop = oplist(a.drop); move = oplist(a.move); cpuw = oplist(a.cpu_weights)
dropped_cmds = set()
for e in main + work:
    c = cmds[e[0]]
    if (c[1], c[2]) in drop:
        assert not c[4] and not c[5] and not c[6] and not e[4], ('not a no-op', e[0], c[1], c[2])
        dropped_cmds.add(e[0])
for e in main:
    c = cmds[e[0]]
    if (c[1], c[2]) in move and not e[4]:
        lane[e[0]] = 1
qlane = {e[0]: (0 if k == 0 else 1) for k, lst in enumerate((main, work)) for e in lst}
kept_main = [e for e in main if e[0] not in dropped_cmds and lane[e[0]] == 0]
print('dropped', len(dropped_cmds), 'moved off main', sum(1 for e in main if lane[e[0]] == 1), 'main entries', len(kept_main))
# --- edge set
succ = [set() for _ in range(N)]
def add(p, i): succ[p].add(i)
if a.free_main:
    # Only the descriptor queue order is fixed: copies, checkpoint sets and waits.
    dq = [e for e in kept_main if e[4] or (cmds[e[0]][1] == 4 and (cmds[e[0]][1], cmds[e[0]][2]) not in cpuw)]
    for x, y in zip(dq, dq[1:]): add(x[0], y[0])
    print('free main: ordered DMA-queue entries', len(dq))
else:
    for x, y in zip(kept_main, kept_main[1:]): add(x[0], y[0])
honored = dropped = extra = rejected = 0; pending = []
for i in range(N):
    for p in preds[i]:
        if hb(p, i): add(p, i); honored += 1
        elif hb(i, p): add(i, p); dropped += 1  # keep QNN's order for pairs it runs against program order
        else: pending.append((p, i))
def reaches(s, t):
    seen = {s}; st = [s]
    while st:
        u = st.pop()
        if u == t: return True
        for v in succ[u]:
            if v not in seen: seen.add(v); st.append(v)
    return False
unordered = len(pending)
if a.strict:
    for p, i in pending:
        if reaches(i, p): rejected += 1
        else: add(p, i); extra += 1
print(f'edges: honored {honored}, QNN-reversed kept {dropped}, unordered-in-QNN {unordered} (enforced {extra}, cycle-rejected {rejected})')

# --- async weight copies land only at their later checkpoint wait
mainpos = {e[0]: k for k, e in enumerate(main)}
dma_edges = dma_skip = 0
for k, e in enumerate(main):
    c = cmds[e[0]]
    if c[1] != 4 or a.no_dma_edges or (c[1], c[2]) in cpuw: continue
    setk = next((j for j in range(k, len(main)) if main[j][4] == 1), None)
    if setk is None: continue
    ev = main[setk][3]
    waitk = next((j for j in range(setk, len(main)) if main[j][4] == 2 and main[j][3] == ev), None)
    if waitk is None: continue
    wc = main[waitk][0]
    for i in list(succ[e[0]]):
        if i == wc: continue
        if a.free_main or lane[i] != 0:
            # Keep exactly the landing order QNN guarantees for this consumer.
            if hb(wc, i): add(wc, i); dma_edges += 1
            else: dma_skip += 1
print(f'DMA wait edges added {dma_edges}, not ordered by QNN {dma_skip}')
if a.serial:
    # A chain along a topological order of the existing edges.
    d_ = [0] * N
    for u in range(N):
        for v in succ[u]: d_[v] += 1
    q_ = [i for i in range(N) if d_[i] == 0]; heapq.heapify(q_); order_s = []
    while q_:
        u = heapq.heappop(q_); order_s.append(u)
        for v in succ[u]:
            d_[v] -= 1
            if d_[v] == 0: heapq.heappush(q_, v)
    assert len(order_s) == N
    kept = [i for i in order_s if i not in dropped_cmds]
    for x, y in zip(kept, kept[1:]): add(x, y)
    print('serial chain over', len(kept))
# --- topological order and critical path
indeg = [0] * N
for u in range(N):
    for v in succ[u]: indeg[v] += 1
d = list(indeg); topo = [i for i in range(N) if d[i] == 0]; q = list(topo)
while q:
    u = q.pop()
    for v in succ[u]:
        d[v] -= 1
        if d[v] == 0: topo.append(v); q.append(v)
assert len(topo) == N, 'cycle'
rem = [0.0] * N
for u in reversed(topo): rem[u] = cost[u] + max((rem[v] for v in succ[u]), default=0.0)
for i in dropped_cmds: cost[i] = 0.0
print('critical path %.3f ms, main busy %.3f, worker busy %.3f' % (max(rem), sum(cost[e[0]] for e in kept_main), sum(cost[i] for i in range(N) if lane[i] != 0)))

# Critical-path composition.
_i = max(range(N), key=lambda i: rem[i]); _agg = Counter(); _n = Counter()
while _i is not None:
    _k = ('M' if lane[_i] == 0 else 'W', cmds[_i][1], cmds[_i][2]); _agg[_k] += cost[_i]; _n[_k] += 1
    _i = max(succ[_i], key=lambda v: rem[v], default=None)
print('path:', [(k, round(v, 3), _n[k]) for k, v in _agg.most_common(12)])
if a.model_split:
    # Model only: H on main (in order), M on any of nw+1 other threads, one thread per merge.
    MS = 0.42
    isconv = lambda i: cmds[i][1] == 6 and cmds[i][2] in (32, 33, 36)
    H = {i: cost[i] * (1 - MS) for i in range(N) if isconv(i)}
    Mc = {i: cost[i] * MS * a.model_split for i in H}
    rem2H = {}; rem2M = {}
    order_ = list(reversed(topo))
    for u in order_:
        tail = max((rem2H.get(v, 0.0) for v in succ[u]), default=0.0)
        if u in H:
            rem2M[u] = Mc[u] + max((rem2H[v] for v in succ[u] if not (lane[v] == 0 and v in succ[u] and False)), default=0.0)
            rem2H[u] = H[u] + rem2M[u]
        else:
            rem2H[u] = cost[u] + tail
    d2 = list(indeg); rdy = []; mq = []
    def push(i):
        if lane[i] == 0: heapq.heappush(mq, (-rem2H[i], i))
        else: heapq.heappush(rdy, (-rem2H[i], i))
    for i in range(N):
        if d2[i] == 0: push(i)
    # successors of a split conv: main chain successor may start after H; memory successors after M.
    chain = {}
    for x, y in zip(kept_main, kept_main[1:]): chain[x[0]] = y[0]
    realmem = {u: {v for v in succ[u] if v != chain.get(u) or v in preds_set[u]} for u in range(N)} if False else None
    fr = list(range(1, a.workers + 2)); ev = []; t = 0.0; done = 0; mfree = True; cnt = 0; busy = Counter(); tasks = N + len(H)
    MEMSUCC = [set(x) for x in succ]
    for x, y in chain.items():
        if x in H and y not in [j for j in succ[x] if x in preds[j]]: MEMSUCC[x].discard(y)
    while done < tasks:
        if mfree and mq:
            i = heapq.heappop(mq)[1]; mfree = False; dur_ = H.get(i, cost[i]); heapq.heappush(ev, (t + dur_, cnt, i, 'H' if i in H else 'X', 0)); cnt += 1; busy[0] += dur_
        while fr and rdy:
            _, i = heapq.heappop(rdy)
            l = fr.pop()
            if isinstance(i, tuple): dur_ = Mc[i[0]]
            else: dur_ = cost[i]
            heapq.heappush(ev, (t + dur_, cnt, i, 'W', l)); cnt += 1; busy[l] += dur_
        t, _, i, kind, l = heapq.heappop(ev); done += 1
        if l == 0: mfree = True
        else: fr.append(l)
        def rel(v):
            d2[v] -= 1
            if d2[v] == 0: push(v)
        if kind == 'H':
            heapq.heappush(rdy, (-rem2M[i], (i, 'M')))
            c_ = chain.get(i)
            if c_ is not None and c_ not in MEMSUCC[i]: rel(c_)
        elif kind == 'W' and isinstance(i, tuple):
            for v in MEMSUCC[i[0]]: rel(v)
        else:
            for v in succ[i]: rel(v)
    print('split-merge model (M x%.2f): end %.3f ms; busy' % (a.model_split, t), {k: round(v, 2) for k, v in sorted(busy.items())})
    sys.exit(0)
# --- list schedule: main fixed, worker lanes by longest remaining path
nw = a.workers
STEAL = a.main_steal_us / 1000.0
def schedule(pri):
    d = list(indeg); ready = []; mhead = []; seq = [[] for _ in range(nw + 1)]; start = [0.0] * N
    def push(i): heapq.heappush(mhead if lane[i] == 0 else ready, (-pri[i], i))
    for i in range(N):
        if d[i] == 0: push(i)
    free = list(range(1, nw + 1)); ev = []; now = 0.0; mfree = True; done = 0
    while done < N:
        if mfree and mhead:
            i = heapq.heappop(mhead)[1]; mfree = False; heapq.heappush(ev, (now + cost[i], i, 0)); seq[0].append(i); start[i] = now
        while free and ready:
            _, i = heapq.heappop(ready); l = min(free); free.remove(l)
            heapq.heappush(ev, (now + cost[i], i, l)); seq[l].append(i); start[i] = now
        if mfree and ready and STEAL:
            # all worker lanes are busy and main has nothing: take the best short ready command
            pick = None; held = []
            while ready:
                x = heapq.heappop(ready)
                if cost[x[1]] <= STEAL: pick = x; break
                held.append(x)
            for x in held: heapq.heappush(ready, x)
            if pick:
                i = pick[1]; mfree = False; heapq.heappush(ev, (now + cost[i], i, 0)); seq[0].append(i); start[i] = now
        t, i, l = heapq.heappop(ev); now = t; done += 1
        if l == 0: mfree = True
        else: free.append(l)
        for v in succ[i]:
            d[v] -= 1
            if d[v] == 0: push(v)
    return now, seq, start
now, seq, start = schedule(rem)
print('greedy modeled end %.3f' % now)
if a.search:
    rng = random.Random(7); best = (now, seq, start); pri = rem
    for trial in range(a.search):
        amp = rng.choice((0.002, 0.005, 0.01, 0.02, 0.05))
        p2 = [x * (1 + rng.uniform(-amp, amp)) + rng.uniform(0, amp) for x in rem]
        r = schedule(p2)
        if r[0] < best[0]: best = r
    now, seq, start = best
    print('search best modeled end %.3f' % now)
assert a.free_main or STEAL or [x[0] for x in kept_main] == [i for i in seq[0] if i not in dropped_cmds]
print('modeled end %.3f ms; lane busy' % now, [round(sum(cost[i] for i in s), 3) for s in seq], 'counts', [len(s) for s in seq])

# --- per-lane wait thresholds. A dropped command has no memory and no effect, but it can
# carry main-chain or conflict edges only if it had a footprint, and it has none.
for s_ in seq: s_[:] = [i for i in s_ if i not in dropped_cmds]
where = {}
for L, s in enumerate(seq):
    for k, i in enumerate(s): where[i] = (L, k)
pr = [set() for _ in range(N)]
for u in range(N):
    if u in dropped_cmds: continue
    for v in succ[u]:
        if v not in dropped_cmds: pr[v].add(u)
L_N = nw + 1
waits = []  # per lane, per position: thresholds for lanes (L+1..L+nw)%L_N
nwaits = 0
for L, s in enumerate(seq):
    row = []
    for k, i in enumerate(s):
        need = [0] * L_N
        for p in pr[i]:
            pl, pk = where[p]
            if pl == L: assert pk < k, 'same-lane order violated'
            else: need[pl] = max(need[pl], pk + 1)
        row.append([need[(L + j + 1) % L_N] for j in range(nw)])
        nwaits += any(row[-1])
    waits.append(row)
print('entries with a real wait:', nwaits)
# Prove the tables cannot deadlock, and that they imply every edge.
pos = [0] * L_N; steps = 0
while steps < N - len(dropped_cmds):
    moved = False
    for L in range(L_N):
        while pos[L] < len(seq[L]) and all(pos[(L + j + 1) % L_N] >= w for j, w in enumerate(waits[L][pos[L]])):
            pos[L] += 1; steps += 1; moved = True
    assert moved, 'deadlock'
dma = {e[0]: (e[3], e[4]) for e in main}
# Check every kept edge is implied by the published tables.
for L, s_ in enumerate(seq):
    for k, i in enumerate(s_):
        for p in pr[i]:
            pl, pk = where[p]
            if pl != L:
                j = (pl - L - 1) % L_N
                assert waits[L][k][j] >= pk + 1, 'edge lost'

with open(a.out, 'w') as f:
    f.write('/* Generated by bench/lanes/gen_qnn_lanes.py; do not hand edit. */\n#ifndef QNN_LANES_PRIMARY_H\n#define QNN_LANES_PRIMARY_H\n#include <stdint.h>\n')
    f.write(f'#define QD_QNN_LANE_COUNT {L_N}u\n#define QD_QNN_LANES_DROPPED {len(dropped_cmds)}u\n')
    f.write('typedef struct { uint16_t command, wait[%d], dma_event_id; uint8_t dma_event_kind, publish; } Nprg2QnnLaneEntry;\n' % nw)
    # A list publishes progress only at positions another list waits for: a waiter's threshold
    # is always an exact position, and earlier work is covered by that later publish.
    needed = [set() for _ in range(L_N)]
    for L_ in range(L_N):
        for row in waits[L_]:
            for j, w in enumerate(row):
                if w: needed[(L_ + j + 1) % L_N].add(w)
    print('publish points', [len(x) for x in needed], 'of', [len(x) for x in seq])
    for L, s in enumerate(seq):
        f.write(f'#define QD_QNN_LANE{L}_COUNT {len(s)}u\nstatic const Nprg2QnnLaneEntry nprg2_qnn_lane{L}[{len(s)}] = {{\n')
        for k, i in enumerate(s):
            di, dk = dma.get(i, (0, 0))
            f.write('  {%d,{%s},%d,%d,%d},\n' % (i, ','.join(map(str, waits[L][k])), di, dk, 1 if k + 1 in needed[L] else 0))
        f.write('};\n')
    f.write('#endif\n')

def write_events(path, order):
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 2, 0, 0))
        for i in order:
            di, dk = dma.get(i, (0, 0))
            f.write(struct.pack('<6I', i, 0, 0, qlane[i], di, dk))
def topo_by(key):
    d = list(indeg); q = []; out = []
    for i in range(N):
        if d[i] == 0: heapq.heappush(q, (key(i), i))
    while q:
        _, u = heapq.heappop(q); out.append(u)
        for v in succ[u]:
            d[v] -= 1
            if d[v] == 0: heapq.heappush(q, (key(v), v))
    assert len(out) == N
    return out
if a.events:
    os.makedirs(a.events, exist_ok=True)
    with open(a.events + '/spec.txt', 'w') as f:
        for i in range(N):
            f.write(str(i) + ''.join(f' {p}:{b}' for p, b in sorted(x for x in (R[i] | W[i]) if x[0] != 'T')) + '\n')
    write_events(a.events + '/index.bin', list(range(N)))
    write_events(a.events + '/sched.bin', sorted(range(N), key=lambda i: (start[i], i)))
    write_events(a.events + '/latest.bin', topo_by(lambda i: -i))
    write_events(a.events + '/earliest.bin', topo_by(lambda i: i))
    for seed in (1, 2, 3):
        rng = random.Random(seed); pri = [rng.random() for _ in range(N)]
        write_events(a.events + f'/random{seed}.bin', topo_by(lambda i: pri[i]))
    # Worker commands as late as possible relative to main, then as early as possible.
    write_events(a.events + '/mainfirst.bin', topo_by(lambda i: (lane[i], -i)))
    write_events(a.events + '/workerfirst.bin', topo_by(lambda i: (1 - lane[i], -i)))
    print('events in', a.events)
