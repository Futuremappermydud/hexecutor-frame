#!/usr/bin/env python3
"""Per-operation oracle for QAIRT 2.50 FP16 HTP contexts from the x86 simulator.

Inputs are a context decoded by depth-model/tools/htp_graph.py (graph.json) and
a trace directory written by depth-model/sim/qnn_trace_hook.cpp with
QD_TRACE_LOG and QD_WRITES. The tool maps every executed run-list entry to its
serialized operation, replays the VTCM writes in execution order and decodes
any VTCM tensor before or after the operation that uses it. Native FP16
handlers are checked against these values. Offline development only; it reads
no device and runs no model.

Run-list mapping for 2.50 FP16 graphs: list 1 holds the flags-4/6 operations
in serial order. List 0 holds every other executable operation (not Const,
Shape or @Dummy*) in serial order, plus executor-only InputSlicePad stub
entries, found by alignment against known function offsets.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

BLOCK = 2048
VTCM_BYTES = 8 << 20
PAD_STUB = "0xebc680"
DTYPES = {"uint8": np.uint8, "int8": np.int8, "uint16": np.uint16, "int16": np.int16,
          "int32": np.int32, "float32": np.float32, "int32_quantized": np.int32}
# Element size 2 with a float class letter is binary16 (class names: C/c crouton,
# F/f flat, X/x weights; e/E element size 2, h/f/i suffix for type).
FP16_CLASSES = {"Ce", "ce", "Fe", "fe"}


def load(path):
    return json.loads(Path(path).read_text())


def executable(node):
    t = node["type_name"]
    return not (t in ("Const", "Shape") or t.startswith("@Dummy"))


def read_trace(trace_dir):
    rows = []
    for line in (Path(trace_dir) / "trace.log").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        seq, lst, idx, fn, _tid, n = line.split()
        rows.append((int(seq), int(lst), int(idx), fn, int(n)))
    rows.sort()
    return rows


def _align_worker(work_rows, worker):
    W, V = len(work_rows), len(worker)
    if W == V:
        return {r[2]: n for r, n in zip(work_rows, worker)}
    if W < V:
        raise ValueError(f"worker list has {W} entries, graph {V}")

    extra = W - V
    valid_splits = []
    for k in range(V):
        if all(work_rows[k + i][3] == work_rows[k][3] for i in range(extra + 1)):
            types = defaultdict(set)
            for r_idx, r in enumerate(work_rows):
                if k <= r_idx <= k + extra:
                    n = worker[k]
                elif r_idx > k + extra:
                    n = worker[r_idx - extra]
                else:
                    n = worker[r_idx]
                types[r[3]].add(n["type_name"].split("@")[0])
            if not any(len(t) > 1 for t in types.values()):
                valid_splits.append(k)

    if len(valid_splits) == 1:
        k = valid_splits[0]
        out = {}
        for r_idx, r in enumerate(work_rows):
            if k <= r_idx <= k + extra:
                out[r[2]] = worker[k]
            elif r_idx > k + extra:
                out[r[2]] = worker[r_idx - extra]
            else:
                out[r[2]] = worker[r_idx]
        return out

    raise ValueError(f"worker list has {W} entries, graph {V}: ambiguous or unhandled split (candidates: {valid_splits})")


def map_entries(graph, rows, known=None):
    """Return {(list, idx): node or None for executor-only entries}."""
    nodes = graph["nodes"]
    main_rows = sorted((r for r in rows if r[1] == 0), key=lambda r: r[2])
    work_rows = sorted((r for r in rows if r[1] == 1), key=lambda r: r[2])
    worker = [n for n in nodes if n["flags"] in (4, 6)]
    main = [n for n in nodes if n["flags"] not in (4, 6) and executable(n)]
    worker_map = _align_worker(work_rows, worker)
    known = known or {}

    # Align main rows to main nodes; only rows may be skipped (executor-only).
    A, B = len(main_rows), len(main)
    if A < B:
        raise ValueError(f"main list has {A} entries for {B} operations")
    inf = float("inf")
    cost = np.full((A + 1, B + 1), inf)
    cost[0, 0] = 0
    move = np.zeros((A + 1, B + 1), np.int8)

    def score(i, j):
        fn, t = main_rows[i][3], main[j]["type_name"]
        if fn == PAD_STUB:
            return 100
        if fn in known:
            return 0 if t in known[fn] else 100
        return 1

    for i in range(A + 1):
        for j in range(max(0, B - (A - i)), min(i, B) + 1):
            if i and cost[i - 1, j] + 20 < cost[i, j]:
                cost[i, j] = cost[i - 1, j] + 20
                move[i, j] = 1
            if i and j:
                c = cost[i - 1, j - 1] + score(i - 1, j - 1)
                if c < cost[i, j]:
                    cost[i, j] = c
                    move[i, j] = 2
    out = {}
    i, j = A, B
    while i:
        if move[i, j] == 2:
            out[(0, main_rows[i - 1][2])] = main[j - 1]
            i, j = i - 1, j - 1
        else:
            if main_rows[i - 1][3] != PAD_STUB:
                raise ValueError(f"unmatched main entry {main_rows[i - 1]}")
            out[(0, main_rows[i - 1][2])] = None
            i -= 1
    for r in work_rows:
        out[(1, r[2])] = worker_map[r[2]]
    # Every function offset must name one kernel type.
    types = defaultdict(set)
    for r in rows:
        n = out[(r[1], r[2])]
        if n is not None:
            types[r[3]].add(n["type_name"].split("@")[0])
    bad = {f: sorted(t) for f, t in types.items() if len(t) > 1}
    if bad:
        raise ValueError(f"inconsistent run-list mapping: {bad}")
    return out


def read_writes(trace_dir):
    rec = np.dtype([("seq", "<u4"), ("list", "<u4"), ("idx", "<u4"), ("block", "<u4"),
                    ("data", "u1", BLOCK)])
    return np.fromfile(Path(trace_dir) / "writes.bin", dtype=rec)


def dtype_of(t):
    if t["class_name"] in FP16_CLASSES or (t["dtype"] == 8 and t["element_bytes"] == 2):
        return np.dtype(np.float16)
    if isinstance(t["dtype"], str):
        return np.dtype(DTYPES[t["dtype"]])
    return np.dtype({1: np.uint8, 2: np.uint16, 4: np.uint32}[t["element_bytes"]])


def decode(t, mem):
    """Logical [b, h, w, d] contents of a pool-1 tensor, or None."""
    shape, store = t.get("shape"), t.get("storage") or {}
    if not shape:
        return None
    dt = dtype_of(t)
    dims, pad = shape["dimensions"], shape["padding"]
    # max_dimensions is the whole padded extent; padding locates dims in it.
    full = list(shape["max_dimensions"])
    if store.get("direct"):
        d = store["direct"]
        if d["pool"] != 1:
            return None
        n = int(np.prod(full)) * dt.itemsize
        arr = mem[d["offset"]:d["offset"] + n].view(dt).reshape(full)
        return arr[tuple(slice(p, p + s) for p, s in zip(pad, dims))].copy()
    blocks = store.get("blocks")
    if not blocks or t["class_name"][0] not in "Cc":
        return None
    chunk = (8, 8, 32) if dt.itemsize == 1 else (8, 4, 32)
    nb = [full[0], -(-full[1] // chunk[0]), -(-full[2] // chunk[1]), -(-full[3] // chunk[2])]
    out = np.zeros([nb[0], nb[1] * chunk[0], nb[2] * chunk[1], nb[3] * chunk[2]], dt)
    k = store.get("slice_offset", 0)
    for b in range(nb[0]):
        for bh in range(nb[1]):
            for bw in range(nb[2]):
                for bd in range(nb[3]):
                    ent = blocks[k] if k < len(blocks) else None
                    k += 1
                    if not ent or ent.get("pool") != 1:
                        continue
                    raw = mem[ent["offset"]:ent["offset"] + BLOCK].view(dt)
                    tile = (raw.reshape(8, 8, 32) if dt.itemsize == 1 else
                            raw.reshape(8, 2, 32, 2).transpose(0, 1, 3, 2).reshape(8, 4, 32))
                    out[b, bh * 8:(bh + 1) * 8, bw * chunk[1]:(bw + 1) * chunk[1],
                        bd * 32:(bd + 1) * 32] = tile
    return out[:, pad[1]:pad[1] + dims[1], pad[2]:pad[2] + dims[2], pad[3]:pad[3] + dims[3]].copy()


class Oracle:
    def __init__(self, context_dir, trace_dir, known=None):
        self.graph = load(Path(context_dir) / "graph.json")
        self.tensors = {t["id"]: t for t in self.graph["tensors"]}
        self.rows = read_trace(trace_dir)
        self.nodes = map_entries(self.graph, self.rows, known)
        self.writes = read_writes(trace_dir)

    def run(self, want=None):
        """Yield (row, node, mem_before, mem_after) in execution order.

        mem_before excludes the node's own writes; DMA results land at the
        later @DmaCheckpointWait, as in the simulator.
        """
        mem = np.zeros(VTCM_BYTES, np.uint8)
        pos = 0
        for row in self.rows:
            node = self.nodes[(row[1], row[2])]
            interesting = node is not None and (want is None or want(node))
            before = mem.copy() if interesting else None
            while pos < len(self.writes) and self.writes[pos]["seq"] == row[0]:
                b = int(self.writes[pos]["block"])
                mem[b * BLOCK:(b + 1) * BLOCK] = self.writes[pos]["data"]
                pos += 1
            if interesting:
                yield row, node, before, mem

    def tensor(self, tid, mem):
        return decode(self.tensors[tid], mem)


def block_table(t):
    """Pool-1 offsets of a crouton tensor's blocks in [b][h/8][w/4][d/32] order (None: absent)."""
    shape, store = t["shape"], t.get("storage") or {}
    full = list(shape["max_dimensions"])
    count = full[0] * -(-full[1] // 8) * -(-full[2] // 4) * -(-full[3] // 32)
    blocks = store.get("blocks") or []
    start = store.get("slice_offset", 0)
    out = []
    for k in range(start, start + count):
        ent = blocks[k] if k < len(blocks) else None
        out.append(ent["offset"] if ent and ent.get("pool") == 1 else None)
    return out, full


def gather(mem, offsets):
    out = bytearray()
    for off in offsets:
        out += bytes(BLOCK) if off is None else mem[off:off + BLOCK].tobytes()
    return bytes(out)


def direct_offset(t):
    store = t.get("storage") or {}
    if store.get("direct"):
        return store["direct"]["pool"], store["direct"]["offset"]
    blocks = store.get("blocks") or []
    if len(blocks) == 1 and blocks[0]:
        return blocks[0]["pool"], blocks[0]["offset"]
    return None, None


FIXTURE_MAGIC = b"Q16F"
FIXTURE_CONV1X1 = 1


def conv1x1_cases(o, clipped=False, scalar_t=None):
    """Pointwise ConvLayer.fp16.s1 operations whose packed weights fit the
    32-channel Xpe tile format. ``scalar_t`` optionally selects QNN's t field;
    t=3 is a small-input pointwise case, not a spatial convolution."""
    T = o.tensors
    for row, node, before, after in o.run(want=lambda n: n["type_name"].startswith("ConvLayer.fp16.s1")):
        x, w, fi, t, _cfg = node["inputs"]
        wd, fd = T[w]["shape"]["dimensions"], T[fi]["shape"]["dimensions"]
        qnn_t = T[t].get("scalar")
        if (wd[:2] != [1, 1] or fd[2] not in (1, 2) or
                qnn_t not in (0, 3, 544) or (scalar_t is not None and qnn_t != scalar_t)):
            continue
        if scalar_t is None and (fd[2] == 2) != clipped:
            continue
        a_off, a_full = block_table(T[x])
        c_off, c_full = block_table(T[node["outputs"][0]])
        K, N = -(-a_full[3] // 32), -(-wd[3] // 32)
        M = len(a_off) // K
        if -(-wd[2] // 32) != K or len(c_off) != M * N or fd[1] != N:
            continue
        pw, ow = direct_offset(T[w])
        pf, of = direct_offset(T[fi])
        if pw != 1 or pf != 1:
            continue
        case = {
            "seq": node["sequence"], "M": M, "K": K, "N": N,
            "a": gather(before, a_off),
            "w": before[ow:ow + N * K * BLOCK].tobytes(),
            "T": fd[2],
            "ci": wd[2],
            "fi": before[of:of + N * fd[2] * 256].tobytes(),
            "c": gather(after, c_off),
        }
        case["tol"] = conv1x1_tolerance(case, T[node["outputs"][0]]).tobytes()
        yield case


def tiles_to_pixels(raw, m, groups):
    """[m][groups] FP16 crouton tiles -> [m, 32 pixels, groups * 32 channels]."""
    return (np.frombuffer(raw, np.float16).reshape(m, groups, 16, 32, 2)
            .transpose(0, 2, 4, 1, 3).reshape(m, 32, groups * 32))


def pixels_to_tiles(x, m, groups):
    return x.reshape(m, 16, 2, groups, 32).transpose(0, 3, 1, 4, 2).reshape(-1)


def conv1x1_tolerance(case, out):
    """Per-output-half float32 tolerance in tile order.

    QAIRT's x86 simulator evaluates HMX in exact FP32, the Hexagon ISS (and
    silicon) round inside the accumulator, so a handler matches when it is
    within 2^-10 of sum |a * w * scale| plus one FP16 ULP of the oracle plus
    2^-14 for subnormal outputs.
    Lanes outside the output tensor's dimensions are don't-care (+inf).
    """
    M, K, N = case["M"], case["K"], case["N"]
    a = tiles_to_pixels(case["a"], M, K).astype(np.float64)
    w = (np.frombuffer(case["w"], np.float16).reshape(N, K, 16, 32, 2).transpose(0, 1, 2, 4, 3)
         .reshape(N, K * 32, 32).astype(np.float64))
    # Rows past the packed input-channel extent are whatever VTCM held (they
    # can be NaN bit patterns); they do not contribute to valid outputs.
    w[:, case.get("ci", K * 32):, :] = 0
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    fi = np.frombuffer(case["fi"], np.uint32).reshape(N, case["T"] * 64)
    scale = (fi[:, :32] & 0xffff).astype(np.uint16).view(np.float16).astype(np.float64).reshape(-1)
    w = np.concatenate(list(w), axis=1)
    mag = np.einsum("mpc,cd->mpd", np.nan_to_num(np.abs(a)), np.abs(w)) * np.abs(scale)
    want = tiles_to_pixels(case["c"], M, N).astype(np.float64)
    ulp = np.spacing(np.abs(np.nan_to_num(want)).astype(np.float16)).astype(np.float64)
    # 2^-14 (the smallest normal FP16) absorbs subnormal handling differences.
    tol = mag * 2.0 ** -10 + ulp + 2.0 ** -14
    # Valid lanes: logical [b, h, w, d] positions inside the output's dimensions.
    shape = out["shape"]
    full = list(shape["max_dimensions"])
    lo = shape["padding"]
    hi = [p + d for p, d in zip(lo, shape["dimensions"])]
    valid = np.zeros(tol.shape, bool)
    bh, bw, bd = -(-full[1] // 8), -(-full[2] // 4), -(-full[3] // 32)
    blk = 0
    for b in range(full[0]):
        for y in range(bh):
            for x in range(bw):
                if blk >= M:
                    break
                pix = np.arange(32)
                py, px = y * 8 + pix // 4, x * 4 + pix % 4
                ok = (b >= lo[0]) & (b < hi[0]) & (py >= lo[1]) & (py < hi[1]) & (px >= lo[2]) & (px < hi[2])
                ch = np.arange(N * 32)
                cok = (ch >= lo[3]) & (ch < hi[3])
                valid[blk] = ok[:, None] & cok[None, :]
                blk += 1
    if bd != N:
        raise ValueError("output channel blocks do not match weight groups")
    valid &= np.isfinite(want)
    tol[~valid] = np.inf
    return pixels_to_tiles(tol.astype(np.float32), M, N)


def write_fixture(path, kind, cases):
    import struct
    cases = list(cases)
    with Path(path).open("wb") as f:
        f.write(FIXTURE_MAGIC + struct.pack("<III", 3, kind, len(cases)))
        for c in cases:
            f.write(struct.pack("<IIIII", c["seq"], c["M"], c["K"], c["N"], c["T"]))
            for key in ("a", "w", "fi", "c", "tol"):
                f.write(c[key])
    return len(cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("context", type=Path, help="htp_graph.py output directory")
    p.add_argument("trace", type=Path, help="trace hook output directory")
    p.add_argument("--known", type=Path, help="JSON map of function offset to kernel type names")
    p.add_argument("--census", type=Path, help="write per-kernel census JSON here")
    p.add_argument("--conv1x1", type=Path, help="write linear 1x1 convolution fixtures here")
    p.add_argument("--conv1x1-clip", type=Path, help="write clipped (two-table) 1x1 convolution fixtures here")
    p.add_argument("--conv1x1-small", type=Path, help="write t=3 partial-channel pointwise fixtures here")
    a = p.parse_args()
    known = load(a.known) if a.known else None
    o = Oracle(a.context, a.trace, known)
    if a.conv1x1 or a.conv1x1_clip or a.conv1x1_small:
        if a.conv1x1:
            print(f"{write_fixture(a.conv1x1, FIXTURE_CONV1X1, conv1x1_cases(o))} conv1x1 cases -> {a.conv1x1}")
        if a.conv1x1_clip:
            n = write_fixture(a.conv1x1_clip, FIXTURE_CONV1X1, conv1x1_cases(o, clipped=True))
            print(f"{n} clipped conv1x1 cases -> {a.conv1x1_clip}")
        if a.conv1x1_small:
            # t=3 uses the ordinary one-K-tile Xpe layout with a partial
            # logical input-channel count. t=544 stages two activation tiles
            # for a 4-channel weight tensor and needs its own kernel contract.
            cases = list(conv1x1_cases(o, clipped=True, scalar_t=3))
            n = write_fixture(a.conv1x1_small, FIXTURE_CONV1X1, cases)
            print(f"{n} small-input pointwise cases -> {a.conv1x1_small}")
        return
    kinds = Counter()
    fn = {}
    for row, node, _before, _after in o.run(want=lambda n: True):
        kinds[node["type_name"]] += 1
        fn[node["type_name"]] = row[3]
    summary = {"executed": sum(kinds.values()), "executor_only": sum(1 for v in o.nodes.values() if v is None),
               "kernels": {k: {"count": c, "function": fn[k]} for k, c in kinds.most_common()}}
    if a.census:
        a.census.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("executed", "executor_only")}))


if __name__ == "__main__":
    main()
