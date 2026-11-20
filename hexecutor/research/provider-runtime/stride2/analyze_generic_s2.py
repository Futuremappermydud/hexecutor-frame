#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the raw V69 S2 leaf against canonical and basis-recovered int64 maps."""
from __future__ import annotations

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import hashlib
import json
import struct
import sys
import argparse
from pathlib import Path
import numpy as np

HERE = (PATHS.repo / 'hexecutor/research/provider-runtime/stride2')
sys.path.insert(0, str((PATHS.native / 'kernels/stride2')))
import check_s2_oracle as oracle  # noqa: E402

PAGE = 2048
HEADER = 192
TAPS = ((0, 2), (0, 0), (2, 2), (2, 0), (0, 1), (2, 1), (1, 2), (1, 0), (1, 1))

def requant(acc: np.ndarray, fi_raw: bytes) -> np.ndarray:
    co = acc.shape[2]
    fi = np.frombuffer(fi_raw, dtype="<i4").reshape(co // 32, 2, 32)
    mul = np.empty(co, dtype=np.int64)
    bias = np.empty(co, dtype=np.int64)
    for cb in range(co // 32):
        mul[cb * 32:(cb + 1) * 32] = fi[cb, 0]
        bias[cb * 32:(cb + 1) * 32] = fi[cb, 1]
    out = np.empty(acc.shape, dtype=np.uint8)
    for ch in range(co):
        hword = int(mul[ch]) & 0xffff
        shift = 34 - ((hword >> 10) & 31)
        pre, post = shift - 14, 14
        a = acc[:, :, ch] + int(bias[ch])
        shifted = (a >> pre) if pre >= 0 else (a << -pre)
        q = ((shifted * (1024 + (hword & 1023)) + (1 << (post - 1))) >> post)
        out[:, :, ch] = np.clip(q, 0, 255).astype(np.uint8)
    return out

def canonical_acc(xlogical: np.ndarray, w: np.ndarray, oh: int, ow: int) -> np.ndarray:
    ci, co = w.shape[2:]
    lowered = oracle.lower_s2(oracle.pack_xwb(w), ci, co)
    acc = np.zeros((oh, ow, co), dtype=np.int64)
    for a in range(2):
        for b in range(2):
            patch = xlogical[a:a + oh, b:b + ow, :4 * ci].astype(np.int64)
            acc += patch @ lowered[a, b].astype(np.int64)
    return acc

def decode_tap_map() -> dict[tuple[int, int], tuple[int, int, int]]:
    """Map raw tap -> lowered (spatial y,x, 32-channel page group)."""
    out = {}
    for ky, kx in TAPS:
        found = []
        for a in range(2):
            for b in range(2):
                for dy in range(2):
                    for dx in range(2):
                        if 2 * a + dy == ky and 2 * b - dx == kx:
                            found.append((a, b, 2 * dy + dx))
        if len(found) != 1:
            raise AssertionError((ky, kx, found))
        out[(ky, kx)] = found[0]
    return out

def hmx_addressed_acc(xpages: bytes, w: np.ndarray, max_w: int, input_channels: int,
                      oh: int, ow: int) -> np.ndarray:
    """Scalar oracle using leaf-observed per-tap byte lane: ci + 4*group.

    Input pages are the exact physical tensor pages used by the captured leaf.
    For each tap the leaf's input table page group is 2*dy+dx. Basis runs show
    that the VMEM byte offset within each 32-channel pixel is ci+4*group.
    """
    ci, co = w.shape[2:]
    groups = input_channels // 32
    nbw = (max_w + 7) // 8
    taps = decode_tap_map()
    acc = np.zeros((oh, ow, co), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            # Xwb's raw spatial index k is reordered by the legacy S2
            # lowering: TAPS[k] is the logical 3x3 coordinate consumed by
            # the 2x2 space-to-depth arms.
            logical_ky, logical_kx = TAPS[ky * 3 + kx]
            a, b, group = taps[(logical_ky, logical_kx)]
            for y in range(oh):
                iy = y + a
                for x in range(ow):
                    ix = x + b
                    block = ((iy // 8) * nbw + ix // 8) * groups + group
                    pagebase = block * PAGE
                    pix = (iy % 8) * 256 + (ix % 8) * 32
                    for source_ci in range(ci):
                        laneoff = source_ci + 4 * group
                        xv = xpages[pagebase + pix + laneoff]
                        acc[y, x] += int(xv) * w[ky, kx, source_ci].astype(np.int64)
    return acc

def hist(delta: np.ndarray) -> dict[str, object]:
    keys, vals = np.unique(delta, return_counts=True)
    pairs = sorted(((int(k), int(v)) for k, v in zip(keys, vals)), key=lambda p: (-p[1], p[0]))
    bad = int(np.count_nonzero(delta))
    return {"elements": int(delta.size), "mismatches": bad,
            "exact": int(delta.size - bad), "min_delta": int(delta.min()),
            "max_delta": int(delta.max()), "most_common_deltas": pairs[:12]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", action="store_true", help="read owned synthetic basis fixture")
    args = ap.parse_args()
    fixture_path = (PATHS.repo / 'hexecutor/research/provider-runtime/stride2/s2_basis_fixture.bin') if args.basis else (PATHS.native / 'kernels/stride2/s2_fixture_224.bin')
    fixture = fixture_path.read_bytes()
    h = struct.unpack_from("<48I", fixture)
    if h[1] != 224:
        raise SystemExit("input fixture is not primary sequence 224")
    xpages_n, wb, fb, ypages_n = h[15] * PAGE, h[35], h[36], h[29]
    x0 = HEADER
    w0 = x0 + xpages_n
    fi0 = w0 + wb
    y0 = fi0 + fb
    xpages = fixture[x0:w0]
    wraw = fixture[w0:fi0]
    firaw = fixture[fi0:y0]
    cpu_pages = fixture[y0:y0 + ypages_n * PAGE]
    native_pages = ((PATHS.repo / 'hexecutor/research/provider-runtime/stride2/v69_s2_output_224.bin')).read_bytes()
    if len(native_pages) != ypages_n * PAGE:
        raise SystemExit("missing/incorrect native leaf output dump")

    xd, xm, yd = h[2:6], h[6:10], h[16:20]
    wh, ww, ci, co = h[30:34]
    w = oracle.unpack_xwb(wraw, wh, ww, ci, co)
    xlogical = oracle.read_blocked(xpages, xm[1], xm[2], xm[3], xm[2], xm[3])
    cpu_logical = oracle.read_output(cpu_pages, yd[1], yd[2], co, h[22], h[23])
    got_logical = oracle.read_output(native_pages, yd[1], yd[2], co, h[22], h[23])
    can = requant(canonical_acc(xlogical, w, yd[1], yd[2]), firaw)
    hmx = requant(hmx_addressed_acc(xpages, w, xm[2], xm[3], yd[1], yd[2]), firaw)
    active_pages = ((yd[1] + 7) // 8) * ((yd[2] + 7) // 8) * (co // 32)
    report = {
        "sequence": 224,
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "native_output_sha256": hashlib.sha256(native_pages).hexdigest(),
        "captured_leaf_roles": {"r0": "output request/table", "r1": "input descriptor/table", "r2": "raw Xwb", "r3": "raw FI", "r5": xm[3]},
        "captured_descriptor_input": ["table", xm[3] // 32, (xm[2] + 7) // 8 * (xm[3] // 32), xm[2], xm[1], xm[3]],
        "captured_descriptor_output": ["table", co // 32, (yd[2] + 7) // 8 * (co // 32), yd[2], yd[1], co],
        "native_pages": ypages_n,
        "descriptor_active_pages": active_pages,
        "canonical_s2_oracle_matches_cpu": bool(np.array_equal(can, cpu_logical)),
        "canonical_s2_oracle_vs_leaf_hist_active": hist(got_logical.astype(np.int16) - can.astype(np.int16)),
        "basis_lane_map_oracle_vs_cpu_hist_active": hist(hmx.astype(np.int16) - cpu_logical.astype(np.int16)),
        "basis_lane_map_oracle_vs_leaf_hist_active": hist(got_logical.astype(np.int16) - hmx.astype(np.int16)),
        "basis_tap_map": {f"{ky},{kx}": list(v) for (ky, kx), v in decode_tap_map().items()},
    }
    ((PATHS.repo / 'hexecutor/research/provider-runtime/stride2/raw_s2_numeric_audit.json')).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
