#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Remove channels that cannot affect any output from a lowered HXFG graph.

Pruned models keep their full channel width in the file. A channel is removable
when, in every tensor it passes through, it is exactly zero or never read. Graph
inputs and outputs, and any tensor a shape-changing operator touches, keep every
channel and their order.

The remaining internal channels are ordered by descending total weight magnitude
in the convolutions that read them. HMX accumulation is order-sensitive; on the
validated MediaPipe landmark graph this reduced the worst handedness error from
0.0205 to 0.0191. Removal alone is bit-identical on the FP32 reference; reordering
changes FP32 summation order, so validate the result against the source model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

NONE = 0xFFFFFFFF
CONV, DEPTHWISE, ADD, PRELU, MAXPOOL, PAD, RESIZE, RESHAPE, CONCAT, MEAN, LOGISTIC = range(1, 12)
TIED = (DEPTHWISE, ADD, PRELU, MAXPOOL, RESIZE, MEAN)


def aligned(n):
    return (n + 127) & ~127


def read(prefix):
    b = prefix.with_suffix(".hxfg").read_bytes()
    _, _, size, nt, nop, ni, no, _, _, to, oo, io = struct.unpack_from("<12I", b)
    if size != len(b):
        raise ValueError("incorrect image size")
    tensors = [list(struct.unpack_from("<8I", b, to + i * 32)) for i in range(nt)]
    ops = [list(struct.unpack_from("<16I", b, oo + i * 64)) for i in range(nop)]
    ids = struct.unpack_from(f"<{ni + no}I", b, io)
    return tensors, ops, list(ids[:ni]), list(ids[ni:]), prefix.with_suffix(".bin").read_bytes()


def conv_weights(pool, op, tensors):
    """FP32 weights as [taps * input channels, output stride]."""
    a, b = tensors[op[1]], tensors[op[4]]
    k = op[5] * op[6] * a[3]
    return np.frombuffer(pool, "<f4", k * b[4], op[11]).reshape(k, b[4])


def analyse(tensors, ops, inputs, outputs, pool):
    """Returns the channels kept per tensor."""
    nt = len(tensors)
    parent = list(range(nt))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pinned = set(inputs) | set(outputs)
    for op in ops:
        kind, x, y, _, out = op[:5]
        if kind in TIED:
            parent[root(x)] = root(out)
            if y != NONE:
                parent[root(y)] = root(out)
        elif kind != CONV:
            pinned.update(i for i in (x, y, out) if i != NONE)
    for op in ops:
        # A spatial convolution indexes its input channels by tap.
        if op[0] == CONV and op[5] * op[6] != 1:
            pinned.add(op[1])

    zero = [np.zeros(t[3], bool) for t in tensors]
    used = [np.ones(t[3], bool) if i in pinned else np.zeros(t[3], bool) for i, t in enumerate(tensors)]
    for _ in range(4):
        for op in ops:
            kind, x, y, _, out = op[:5]
            co = tensors[out][3]
            if kind == CONV:
                w = conv_weights(pool, op, tensors) != 0
                bias = np.frombuffer(pool, "<f4", tensors[out][4], op[12])[:co] != 0
                live = ~np.repeat(zero[x], op[5] * op[6])
                zero[out] = ~bias & ~(w[:, :co][live].any(0))
            elif kind == DEPTHWISE:
                taps = op[5] * op[6]
                w = np.frombuffer(pool, "<f4", taps * tensors[out][4], op[11]).reshape(taps, -1)
                bias = np.frombuffer(pool, "<f4", tensors[out][4], op[12])[:co] != 0
                zero[out] = ~bias & (zero[x] | ~w[:, :co].any(0))
            elif kind == ADD:
                zero[out] = zero[x] & zero[y]
            elif kind in (PRELU, MAXPOOL, RESIZE, MEAN):
                zero[out] = zero[x].copy()
        for op in reversed(ops):
            kind, x, y, _, out = op[:5]
            co = tensors[out][3]
            needed = used[out] & ~zero[out]
            if kind == CONV:
                w = conv_weights(pool, op, tensors) != 0
                reads = w[:, :co][:, needed].any(1)
                if op[5] * op[6] == 1:
                    used[x] = used[x] | reads
            elif kind == DEPTHWISE:
                taps = op[5] * op[6]
                w = np.frombuffer(pool, "<f4", taps * tensors[out][4], op[11]).reshape(taps, -1)
                used[x] = used[x] | (needed & w[:, :co].any(0))
            elif kind == ADD:
                used[x] = used[x] | needed
                used[y] = used[y] | needed
            elif kind in (PRELU, MAXPOOL, RESIZE, MEAN):
                used[x] = used[x] | needed
    # A channel survives if any tensor of its tied group still needs its value.
    groups = {}
    for i, t in enumerate(tensors):
        groups.setdefault(root(i), []).append(i)
    keep = [None] * nt
    for members in groups.values():
        c = tensors[members[0]][3]
        if any(i in pinned for i in members):
            mask = np.ones(c, bool)
        else:
            mask = np.zeros(c, bool)
            for i in members:
                mask |= used[i] & ~zero[i]
            if not mask.any():
                mask[0] = True
        kept = np.flatnonzero(mask)
        if not any(i in pinned for i in members):
            score = np.zeros(c)
            for op in ops:
                if op[0] == CONV and op[1] in members and op[5] * op[6] == 1:
                    score += np.abs(conv_weights(pool, op, tensors)).sum(1)
            kept = kept[np.argsort(score[kept], kind="stable")[::-1]]
        for i in members:
            keep[i] = kept
    return keep


def plan_arena(tensors, ops, inputs, outputs):
    last = {}
    for j, op in enumerate(ops):
        for i in op[1:4]:
            if i != NONE:
                last[i] = j
    for i in outputs:
        last[i] = len(ops)
    free, allocated, peak = [], {}, 0

    def allocate(i):
        nonlocal peak
        t = tensors[i]
        n = aligned(t[1] * t[2] * t[4] * 4)
        choice = next((j for j, (_, size) in enumerate(free) if size >= n), None)
        if choice is None:
            at = peak
            peak += n
        else:
            at, size = free.pop(choice)
            if size > n:
                free.append((at + n, size - n))
        allocated[i] = (at, n)
        t[0] = at

    for i in inputs:
        allocate(i)
    for j, op in enumerate(ops):
        for i in list(allocated):
            if last.get(i, -1) < j:
                free.append(allocated.pop(i))
        free.sort()
        merged = []
        for at, n in free:
            if merged and sum(merged[-1]) == at:
                merged[-1] = (merged[-1][0], merged[-1][1] + n)
            else:
                merged.append((at, n))
        free[:] = merged
        allocate(op[4])
    return peak


def compact(source, destination):
    tensors, ops, inputs, outputs, pool = read(source)
    keep = analyse(tensors, ops, inputs, outputs, pool)
    new_pool = bytearray()

    def emit(array, dtype="<f4"):
        at = aligned(len(new_pool))
        new_pool.extend(bytes(at - len(new_pool)))
        new_pool.extend(np.asarray(array, dtype=dtype).tobytes())
        return at

    def stride(i):
        t = tensors[i]
        return t[4] if t[4] == 3 else (len(keep[i]) + 31) & ~31

    def channels(values, i):
        out = np.zeros(stride(i), np.float32)
        out[: len(keep[i])] = values[keep[i]]
        return out

    new_ops = []
    for op in ops:
        kind, x, _, _, out = op[:5]
        op = list(op)
        if kind == CONV:
            co_old = tensors[out][4]
            w = conv_weights(pool, op, tensors).reshape(op[5], op[6], tensors[x][3], co_old)
            if op[5] * op[6] == 1:
                w = w[:, :, keep[x], :]
            w = w[:, :, :, keep[out]]
            w = np.pad(w, ((0, 0), (0, 0), (0, 0), (0, stride(out) - w.shape[3])))
            bias = np.frombuffer(pool, "<f4", co_old, op[12])
            op[11], op[12] = emit(w), emit(channels(bias, out))
            matrix = w.reshape(-1, stride(out))
            k, n = matrix.shape
            kp = (k + 31) & ~31
            matrix = np.pad(matrix, ((0, kp - k), (0, 0)))
            packed = matrix.reshape(kp // 32, 16, 2, n // 32, 32).transpose(3, 0, 1, 4, 2)
            op[14], op[15] = emit(packed, "<f2"), kp // 32
        elif kind == DEPTHWISE:
            s = tensors[out][4]
            w = np.frombuffer(pool, "<f4", op[5] * op[6] * s, op[11]).reshape(op[5] * op[6], s)
            w = w[:, keep[out]]
            w = np.pad(w, ((0, 0), (0, stride(out) - w.shape[1])))
            bias = np.frombuffer(pool, "<f4", s, op[12])
            op[11], op[12] = emit(w), emit(channels(bias, out))
        elif kind == PRELU:
            s = np.frombuffer(pool, "<f4", tensors[x][4], op[5])
            op[5] = emit(channels(s, x))
        new_ops.append(op)
    removed = sum(t[3] - len(k) for t, k in zip(tensors, keep))
    new_tensors = [list(t) for t in tensors]
    for i, t in enumerate(new_tensors):
        t[3], t[4] = len(keep[i]), stride(i)
    peak = plan_arena(new_tensors, new_ops, inputs, outputs)
    to = 256
    oo = to + len(new_tensors) * 32
    io_at = oo + len(new_ops) * 64
    header = (
        struct.pack("<12I", 0x47465848, 1, io_at + (len(inputs) + len(outputs)) * 4, len(new_tensors),
                    len(new_ops), len(inputs), len(outputs), peak, len(new_pool), to, oo, io_at)
        + hashlib.sha256(new_pool).digest()
        + bytes(176)
    )
    body = b"".join(struct.pack("<8I", *t) for t in new_tensors)
    body += b"".join(struct.pack("<16I", *op) for op in new_ops)
    body += struct.pack(f"<{len(inputs) + len(outputs)}I", *(inputs + outputs))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".hxfg").write_bytes(header + body)
    destination.with_suffix(".bin").write_bytes(new_pool)
    manifest_path = source.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(
        arena_bytes=peak,
        constant_bytes=len(new_pool),
        channel_compaction={"source_constant_bytes": len(pool), "channels_removed": removed},
    )
    destination.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="input HXFG prefix")
    p.add_argument("destination", type=Path, help="output prefix, in an ignored artifact directory")
    args = p.parse_args()
    m = compact(args.source, args.destination)
    print(f'{m["channel_compaction"]["channels_removed"]} channels removed, '
          f'{m["arena_bytes"]} arena bytes, {m["constant_bytes"]} constant bytes')


if __name__ == "__main__":
    main()
