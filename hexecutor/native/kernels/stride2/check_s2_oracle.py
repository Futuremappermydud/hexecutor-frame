#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independently lower one actual S2 Xwb case and verify the CPU-prefix output."""

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
import struct
from pathlib import Path
import numpy as np

HERE = (PATHS.native / 'kernels/stride2')
PAGE = 2048
HEADER = 48 * 4
TAPS = ((0, 2), (0, 0), (2, 2), (2, 0), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1))


def unpack_xwb(raw, kh, kw, ci, co):
    out = np.empty((kh, kw, ci, co), dtype=np.int8)
    for h in range(kh):
        for w in range(kw):
            for i in range(ci):
                for o in range(co):
                    off = ((((o // 32) * (ci // 32) + i // 32) * kh + h) * kw + w) * 1024
                    off += ((i // 4) % 8) * 128 + (o % 32) * 4 + i % 4
                    out[h, w, i, o] = np.frombuffer(raw, np.uint8, 1, off)[0].view(np.int8)
    return out


def lower_s2(raw_w, ci, co):
    w = unpack_xwb(raw_w, 3, 3, ci, co)
    tr = np.empty_like(w)
    for k, (y, x) in enumerate(TAPS):
        tr[y, x] = w.reshape(9, ci, co)[k]
    final = np.zeros((2, 2, 4 * ci, co), dtype=np.int8)
    for a in range(2):
        for b in range(2):
            for dy in range(2):
                for dx in range(2):
                    ky, kx = 2 * a + dy, 2 * b - dx
                    if ky < 3 and 0 <= kx < 3:
                        for cb in range(ci // 32):
                            j = cb * 4 + 2 * dy + dx
                            final[a, b, 32 * j:32 * (j + 1)] = tr[ky, kx, 32 * cb:32 * (cb + 1)]
    return final


def pack_xwb(lowered):
    kh, kw, ci, co = lowered.shape
    raw = bytearray(kh * kw * ci * co)
    for h in range(kh):
        for w in range(kw):
            for i in range(ci):
                for o in range(co):
                    off = ((((o // 32) * (ci // 32) + i // 32) * kh + h) * kw + w) * 1024
                    off += ((i // 4) % 8) * 128 + (o % 32) * 4 + i % 4
                    raw[off] = int(lowered[h, w, i, o]) & 255
    return bytes(raw)


def read_blocked(pages, y, x, channels, max_w, max_c):
    nbw, nbd = (max_w + 7) // 8, max_c // 32
    out = np.empty((y, x, channels), dtype=np.uint8)
    for yy in range(y):
        for xx in range(x):
            for cc in range(channels):
                block = ((yy // 8) * nbw + xx // 8) * nbd + cc // 32
                out[yy, xx, cc] = pages[block * PAGE + (yy % 8) * 256 + (xx % 8) * 32 + cc % 32]
    return out


def read_output(pages, y, x, channels, max_w, max_c):
    return read_blocked(pages, y, x, channels, max_w, max_c)


def fnv64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h


def main():
    raw = ((PATHS.native / 'kernels/stride2/s2_cpu_prefix_fixture.bin')).read_bytes()
    h = struct.unpack_from("<48I", raw)
    if h[0] != 0x3253324E:
        raise SystemExit("bad fixture magic")
    # Header positions: x d/m/p at 2/6/10, x slice/count 14/15;
    # y d/m/p at 16/20/24, y slice/count 28/29; W at 30.
    xd, xm, xp = h[2:6], h[6:10], h[10:14]
    x_slice, x_count = h[14:16]
    yd, ym, yp = h[16:20], h[20:24], h[24:28]
    y_slice, y_count = h[28:30]
    wh, ww, ci, co = h[30:34]
    packed, wb, fb, xzp, yzp = h[34:39]
    if ((wh, ww) != (3, 3) or packed != 1 or xzp != 0 or yzp != 0
            or ci % 32 or co % 32 or xd[0] != 1 or yd[0] != 1):
        raise SystemExit("fixture is outside the supported blocked S2 oracle geometry")
    xbytes, ybytes = x_count * PAGE, y_count * PAGE
    xb = raw[HEADER:HEADER + xbytes]
    woff = HEADER + xbytes
    wb_raw = raw[woff:woff + wb]
    fi_off = woff + wb
    fi_raw = raw[fi_off:fi_off + fb]
    yb = raw[fi_off + fb:fi_off + fb + ybytes]
    if len(yb) != ybytes or len(fi_raw) != fb:
        raise SystemExit("truncated fixture")
    # Rebuild the old k_space_to_depth + k_hmx_conv lowered 2x2 operation.
    dense_x = read_blocked(xb[x_slice * PAGE:], xm[1], xm[2], xm[3], xm[2], xm[3])
    lowered = lower_s2(wb_raw, ci, co)
    packed_lowered = pack_xwb(lowered)
    weights = lowered.astype(np.int64)
    fi = np.frombuffer(fi_raw, dtype="<i4").reshape(co // 32, 2, 32)
    multipliers = np.empty(co, dtype=np.int64)
    bias = np.empty(co, dtype=np.int64)
    for cb in range(co // 32):
        multipliers[cb * 32:(cb + 1) * 32] = fi[cb, 0]
        bias[cb * 32:(cb + 1) * 32] = fi[cb, 1]
    acc = np.zeros((yd[1], yd[2], co), dtype=np.int64)
    for ky in range(2):
        for kx in range(2):
            patch = dense_x[ky:ky + yd[1], kx:kx + yd[2], :4 * ci].astype(np.int64)
            acc.reshape(-1, co)[:] += patch.reshape(-1, 4 * ci) @ weights[ky, kx]
    got = np.empty_like(acc, dtype=np.uint8)
    for ch in range(co):
        hword = int(multipliers[ch]) & 0xffff
        shift = 34 - ((hword >> 10) & 31)
        pre, post = shift - 14, 14
        a = acc[:, :, ch] + int(bias[ch])
        q = (((a >> pre) * (1024 + (hword & 1023)) + (1 << (post - 1))) >> post)
        got[:, :, ch] = np.clip(q, 0, 255).astype(np.uint8)
    cpu = read_output(yb[y_slice * PAGE:], yd[1], yd[2], co, ym[2], ym[3])
    delta = got.astype(np.int16) - cpu.astype(np.int16)
    vals, counts = np.unique(delta, return_counts=True)
    hist = {int(k): int(v) for k, v in zip(vals, counts)}
    report = {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "sequence": h[1], "input_dims": list(xd), "input_max": list(xm),
        "input_padding": list(xp), "output_dims": list(yd), "output_max": list(ym),
        "weight_shape": [wh, ww, ci, co], "lowered_shape": [2, 2, 4 * ci, co],
        "cpu_oracle_delta_histogram": hist,
        "lowered_packed_sha256": hashlib.sha256(packed_lowered).hexdigest(),
        "lowered_packed_fnv64": f"{fnv64(packed_lowered):016x}",
        "oracle_exact": bool(np.all(delta == 0)),
        "oracle_abs_le_1": bool(np.all(np.abs(delta) <= 1)),
        "cpu_active_sha256": hashlib.sha256(cpu.tobytes()).hexdigest(),
    }
    import json
    ((PATHS.native / 'kernels/stride2/s2_oracle_report.json')).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["oracle_abs_le_1"]:
        raise SystemExit("independent int64 lower-and-requant oracle differs from CPU output by >1")


if __name__ == "__main__":
    main()
