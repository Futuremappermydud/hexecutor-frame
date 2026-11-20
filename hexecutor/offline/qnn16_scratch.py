#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Give the staged HMX convolutions of a Q16P program VTCM scratch blocks.

Q16_CONV3X3 and Q16_CONV3X3_S2 run on HMX by staging shifted copies of their
input tiles (hexecutor/native/qnn16/qnn16_tile.c). QNN's plan uses the whole
8 MiB VTCM window, so the staging tiles borrow blocks that are dead while the
operation runs: a 2 KiB block is free at an operation when the operation does
not touch it and the next access to it, in program order and wrapping to the
start of the next run, overwrites all of it (a weight COPY, a ZERO, an HMX 1x1
output store, or any operation's output tensor that has no padding and fills
its blocks). Any other access, including a partial write, keeps a block live.
hexecutor/tests/test_qnn16_scratch_poison.py checks the assignment on the host. The block offsets are appended to the word table; p[3] is their word
index and p[4] their count (0: the float path).

With the constant pool, it also marks the Welford partials whose weights are the
reciprocal counts 1 / (i + 1) (p[0] = 1), which lets the DSP use a closed form.

Usage: qnn16_scratch.py IN.q16p OUT.q16p [CONSTANTS.q16c]
"""
import sys
import numpy as np

BLOCK = 2048
VTCM = 0x800000
NONE = 0xFFFFFFFF
K = dict(COPY=1, ZERO=2, CONV1X1=3, DW1X1=4, DW3X3=5, CONV3X3=6, S2D=7, DW3X3_S2=8, IM2COL=9, INPUT=10, OUTPUT=11,
         FORMAT=12, CAST_F32=13, CAST_F16=14, MUL=15, MINMAX=16, SIGMOID=17, RESHAPE=18, CONCAT=19, SLICE=20,
         SHUFFLE=21, TRANSPOSE=22, AVGPOOL=23, X2S=24, ZAP=25, PW1X1=26, SWISH=27, WELFORD_PARTIAL=28,
         CONV3X3_S2=29, WELFORD_MERGE=30, MOVE=31, ADD_SWISH=32, GROUP_MOMENTS=33, DECIMATE=34, SLICE_F32=35,
         INSTANCE_NORM=36, RESIZE_BILINEAR=37, SOFTMAX=38, PW_SWISH=39, REDUCE_SUM=40, RESIZE_NEAREST=41,
         L2NORM=42, REDUCE_MEAN=43, PW_TANH=44, AFFINE=45)
READ_XYZ = {K[n] for n in ("DW1X1", "DW3X3", "CONV3X3", "DW3X3_S2", "CONV3X3_S2", "PW1X1", "PW_SWISH", "PW_TANH")}
READ_X = {K[n] for n in ("S2D", "IM2COL", "OUTPUT", "FORMAT", "CAST_F32", "CAST_F16", "MINMAX", "SIGMOID", "RESHAPE",
                         "SLICE", "SHUFFLE", "TRANSPOSE", "AVGPOOL", "X2S", "SWISH", "MOVE", "ADD_SWISH", "DECIMATE",
                         "SLICE_F32", "RESIZE_BILINEAR", "SOFTMAX", "REDUCE_SUM", "RESIZE_NEAREST",
                         "L2NORM", "REDUCE_MEAN", "AFFINE")}


class Program:
    def __init__(self, image):
        self.image = bytearray(image)
        h = np.frombuffer(image[:128], np.uint32)
        assert h[0] == 0x50363151, "not a Q16P image"
        self.nt, self.nops, self.nwords = int(h[3]), int(h[4]), int(h[5])
        self.toff, self.opoff, self.woff, self.iooff = int(h[8]), int(h[9]), int(h[10]), int(h[11])
        self.tensors = np.frombuffer(image, np.uint32, self.nt * 20, self.toff).reshape(self.nt, 20)
        self.ops = np.frombuffer(image, np.uint32, self.nops * 16, self.opoff).reshape(self.nops, 16).copy()
        self.words = np.frombuffer(image, np.uint32, self.nwords, self.woff).copy()

    def blocks(self, tid):
        """(touched blocks, fully covered blocks) of a tensor in VTCM."""
        t = self.tensors[tid]
        if t[0] == 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        if t[0] == 1:
            lo, hi = int(t[2]), int(t[2]) + int(t[3])
            touched = np.arange(lo // BLOCK, (hi + BLOCK - 1) // BLOCK)
            full = np.arange((lo + BLOCK - 1) // BLOCK, hi // BLOCK)
            return touched, full
        b = self.words[int(t[4]):int(t[4]) + int(t[5])]
        b = (b[b != NONE] // BLOCK).astype(np.int64)
        return b, b

    def accesses(self, i):
        """(blocks read or partly written, blocks wholly overwritten) by operation i."""
        o = self.ops[i]
        kind, x, y, z, out, p = int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), [int(v) for v in o[5:]]
        keep, kill = [], []

        def read(t):
            if t != NONE:
                keep.append(self.blocks(t)[0])

        def write(t):
            # Every handler writes its whole logical output. When the tensor has no padding and, in
            # tiles, fills its blocks (rows, columns and channels in whole blocks), each block it
            # covers is wholly overwritten; an in-place operand stays live through its read.
            if t == NONE:
                return
            touched, full = self.blocks(t)
            row = self.tensors[t]
            whole = list(row[6:10]) == list(row[10:14]) and not any(row[14:18])
            if row[0] == 0:
                whole = whole and not (row[7] % 8 or row[8] % 4 or row[9] % 32)
            if whole:
                kill.append(full)
                keep.append(np.setdiff1d(touched, full))
            else:
                keep.append(touched)

        if kind == K["COPY"]:
            t = self.tensors[out]
            if t[0] == 1:
                lo, hi = int(t[2]), int(t[2]) + p[1]
                full = np.arange((lo + BLOCK - 1) // BLOCK, hi // BLOCK)
                kill.append(full)
                keep.append(np.setdiff1d(np.arange(lo // BLOCK, (hi + BLOCK - 1) // BLOCK), full))
        elif kind == K["ZERO"]:
            touched, full = self.blocks(out)
            kill.append(full)
            keep.append(np.setdiff1d(touched, full))
        elif kind == K["CONV1X1"]:
            m, k, n, at = p[0], p[1], p[2], p[4]
            keep.append(self.words[at:at + m * k].astype(np.int64) // BLOCK)
            kill.append(self.words[at + m * k:at + m * k + m * n].astype(np.int64) // BLOCK)
            read(y), read(z)
        elif kind in READ_XYZ:
            read(x), read(y), read(z), write(out)
            if kind in (K["CONV3X3"], K["CONV3X3_S2"], K["PW1X1"]) and p[4]:
                keep.append(self.words[p[3]:p[3] + p[4]].astype(np.int64) // BLOCK)
        elif kind in READ_X:
            read(x), write(out)
        elif kind == K["INPUT"]:
            write(out)
        elif kind == K["MUL"]:
            read(x), read(y), write(out)
        elif kind == K["CONCAT"]:
            for t in self.words[p[1]:p[1] + p[2]]:
                read(int(t))
            write(out)
        elif kind == K["ZAP"]:
            keep.append(self.words[p[0]:p[0] + p[1]].astype(np.int64) // BLOCK)
        elif kind == K["WELFORD_PARTIAL"]:
            read(x), read(y), write(out)
        elif kind == K["WELFORD_MERGE"]:
            read(x), read(y), write(out)
            for t in self.words[p[0]:p[0] + p[1]]:
                read(int(t))
        elif kind in (K["GROUP_MOMENTS"], K["INSTANCE_NORM"]):
            read(x), read(y), read(z), read(p[0]), write(out)
        else:
            raise SystemExit(f"operation {i}: unknown kind {kind}")
        cat = lambda v: np.unique(np.concatenate(v)) if v else np.zeros(0, np.int64)
        keep, kill = cat(keep), cat(kill)
        return keep, np.setdiff1d(kill, keep)

    def free_before(self):
        """free[i]: boolean per block, usable as scratch while operation i runs."""
        nb = VTCM // BLOCK
        acc = [self.accesses(i) for i in range(self.nops)]
        # state after scanning ops i..end then wrapping: True when the next access is a whole overwrite.
        dead = np.ones(nb, bool)  # a block the program never touches
        for _ in range(2):  # second pass uses the wrapped state
            free = [None] * self.nops
            for i in range(self.nops - 1, -1, -1):
                keep, kill = acc[i]
                dead[kill] = True
                dead[keep] = False
                free[i] = dead.copy()
        for i in range(self.nops):
            keep, kill = acc[i]
            free[i][kill] = False  # the operation's own outputs
            free[i][nb - 1] = False  # HMX completion tile
        return free

    def need(self, i):
        """(staging tiles for every output tile at once, tiles per output tile) of operation i, or (0, 0)."""
        o = self.ops[i]
        kind = int(o[0])
        if kind == K["PW1X1"]:
            # Weight tiles (kg * ng) then temporary output blocks for every output block (ng * blocks).
            out = self.tensors[int(o[4])]
            if out[0] or self.tensors[int(o[1])][0]:
                return 0, 0
            fb, fh, fw, fc = (int(v) for v in out[10:14])
            ng, kg = fc // 32, (int(o[6]) + 31) // 32
            if not ng or not kg:
                return 0, 0
            return kg * ng + ng * fb * ((fh + 7) // 8) * ((fw + 3) // 4), 1
        if kind not in (K["CONV3X3"], K["CONV3X3_S2"]):
            return 0, 0
        x, out = self.tensors[int(o[1])], self.tensors[int(o[4])]
        if x[0] or out[0]:
            return 0, 0
        ob, oh, ow, oc = (int(v) for v in out[6:10])
        fb, fh, fw, fc = (int(v) for v in out[10:14])
        # The output may be a view of fewer rows/columns than its blocks hold, within the same tile grid.
        if (fb, fc) != (ob, oc) or -(-fh // 8) != -(-oh // 8) or -(-fw // 4) != -(-ow // 4) or \
                any(out[14:18]) or oc % 32:
            return 0, 0
        tiles = ob * -(-oh // 8) * -(-ow // 4)  # partial last tiles are staged whole
        groups = (int(o[6]) + 31) // 32 if kind == K["CONV3X3"] else int(x[9]) // 128
        return tiles * groups * 9, groups * 9

    def assign(self):
        free = self.free_before()
        words = list(self.words)
        report = []
        for i in range(self.nops):
            n, per_tile = self.need(i)
            if not n:
                continue
            idx = np.flatnonzero(free[i])
            # Fewer blocks than the whole operation needs: the DSP stages a few output tiles at a time.
            n = min(n, len(idx) // per_tile * per_tile)
            report.append((i, n, per_tile, len(idx)))
            self.ops[i][8] = len(words) if n else 0
            self.ops[i][9] = n
            words.extend(int(b) * BLOCK for b in idx[:n])
        self.words = np.array(words, np.uint32)
        return report

    def mark_welford(self, constants):
        """Set p[0] = 1 on WELFORD_PARTIAL operations whose weights are exactly 1 / (i + 1)."""
        last = {}  # VTCM offset -> (constant offset, bytes) of the latest COPY
        marked = total = 0
        for i in range(self.nops):
            o = self.ops[i]
            kind = int(o[0])
            if kind == K["COPY"]:
                t = self.tensors[int(o[4])]
                if t[0] == 1:
                    last[int(t[2])] = (int(o[5]), int(o[6]))
            elif kind == K["WELFORD_PARTIAL"]:
                total += 1
                x, y = self.tensors[int(o[1])], self.tensors[int(o[2])]
                count, lanes = int(x[7]) * int(x[8]), int(y[9])
                src = last.get(int(y[2])) if y[0] == 1 else None
                o[5] = 0
                if not src or src[1] < count * lanes * 4:
                    continue
                w = np.frombuffer(constants, np.float32, count * lanes, src[0]).reshape(count, lanes)
                want = (np.float32(1) / np.arange(1, count + 1, dtype=np.float32))[:, None]
                if np.array_equal(w, np.broadcast_to(want, w.shape)):
                    o[5] = 1
                    marked += 1
        return marked, total

    def serialize(self):
        h = np.frombuffer(bytes(self.image[:128]), np.uint32).copy()
        io_bytes = (int(h[6]) + int(h[7])) * 32
        tensors = bytes(self.image[self.toff:self.toff + self.nt * 80])
        io = bytes(self.image[self.iooff:self.iooff + io_bytes])
        parts, at = [], 128
        offs = []
        for blob in (tensors, self.ops.tobytes(), self.words.tobytes(), io):
            offs.append(at)
            parts.append(blob)
            at += len(blob)
        h[5] = len(self.words)
        h[8], h[9], h[10], h[11] = offs
        h[2] = at
        return h.tobytes() + b"".join(parts)


def main():
    src, dst = sys.argv[1:3]
    p = Program(open(src, "rb").read())
    for i in range(p.nops):  # drop earlier assignments
        if int(p.ops[i][0]) in (K["CONV3X3"], K["CONV3X3_S2"]):
            p.ops[i][8] = p.ops[i][9] = 0
    report = p.assign()
    none = [r for r in report if not r[1]]
    print(f"{len(report)} staged convolutions, {len(none)} without staging blocks; "
          f"fewest output tiles per pass {min((r[1] // r[2] for r in report), default=0)}")
    for r in none[:20]:
        print(f"  op {r[0]}: needs {r[2]} per output tile, free {r[3]}")
    if len(sys.argv) > 3:
        marked, total = p.mark_welford(open(sys.argv[3], "rb").read())
        print(f"{marked} of {total} Welford partials use reciprocal counts")
    open(dst, "wb").write(p.serialize())


if __name__ == "__main__":
    main()
