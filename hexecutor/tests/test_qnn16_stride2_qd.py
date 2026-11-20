#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay Mercury's stride-2 path against QD writes.

ConvLayer_SpaceToDepth_stride2 must match bit for bit. ConvLayer.fp16.s2
(t=514 depthwise 3x3 over the staged input) must be within the HMX tolerance
2^-10 * sum |x * w * scale| + 1 ULP + 2^-14 used by the other convolution
checks.
"""
import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from qnn16_oracle import Oracle, direct_offset, load  # noqa: E402

TILE_OF_TAP = (1, 4, 0, 7, 8, 6, 3, 5, 2)
OFFSET = (0, 1, 1)
PHASE = (0, 1, 0)


class Shape(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in "bhwc"]


def u16(a):
    return a.ctypes.data_as(C.POINTER(C.c_uint16))


def magnitude(x, w, scale, out_shape):
    """sum |x * w * scale| for every output of the stride-2 depthwise conv."""
    _, oh, ow, oc = out_shape
    mag = np.zeros(out_shape, np.float64)
    lane = np.arange(oc) % 32
    group = np.arange(oc) // 32
    for ky in range(3):
        for kx in range(3):
            q = PHASE[ky] * 2 + (kx == 1)
            ci = (group * 4 + q) * 32 + lane
            wi = (group * 9 + TILE_OF_TAP[ky * 3 + kx]) * 1024 + (lane // 2) * 64 + lane * 2 + lane % 2
            iy, ix = (ky == 2), OFFSET[kx]
            mag[0] += np.abs(x[0, iy:iy + oh, ix:ix + ow][:, :, ci] * w[wi] * scale[:oc])
    return mag


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("context", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--known", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-stride2-") as tmp:
        so = Path(tmp) / "stride2.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        lib = C.CDLL(str(so))
        s2d = lib.qnn16_space_to_depth_stride2_f16
        s2d.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), Shape, C.c_uint32, C.c_uint32]
        conv = lib.qnn16_depthwise3x3_stride2_f16
        conv.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), Shape, C.c_uint32, C.c_uint32,
                         C.POINTER(C.c_uint16), C.c_void_p, C.c_uint32]
        oracle = Oracle(a.context, a.trace, load(a.known))
        T = oracle.tensors
        counts = {"s2d": 0, "conv": 0}
        worst = 0.0
        for _row, node, before, after in oracle.run(want=lambda n: (
                n["type_name"].startswith("ConvLayer_SpaceToDepth_stride2") or
                n["type_name"].startswith("ConvLayer.fp16.s2"))):
            expected = oracle.tensor(node["outputs"][0], after)
            got = np.empty(expected.size, np.uint16)
            if node["type_name"].startswith("ConvLayer_SpaceToDepth"):
                source = np.ascontiguousarray(oracle.tensor(node["inputs"][0], before)).view(np.uint16)
                if s2d(u16(got), u16(source), Shape(*source.shape), expected.shape[1], expected.shape[2]):
                    raise AssertionError(f"space-to-depth op {node['sequence']} rejected")
                if not np.array_equal(got, expected.reshape(-1).view(np.uint16)):
                    bad = np.flatnonzero(got != expected.reshape(-1).view(np.uint16))[0]
                    raise AssertionError(f"space-to-depth op {node['sequence']} differs at "
                                         f"{np.unravel_index(bad, expected.shape)}")
                counts["s2d"] += 1
                continue
            x, w, fi, t, _cfg = node["inputs"]
            if T[t].get("scalar") != 514:
                raise AssertionError(f"stride-2 op {node['sequence']} has t={T[t].get('scalar')}")
            source = np.ascontiguousarray(oracle.tensor(x, before)).view(np.uint16)
            groups = (expected.shape[3] + 31) // 32
            tables = T[fi]["shape"]["dimensions"][2]
            _, woff = direct_offset(T[w])
            _, foff = direct_offset(T[fi])
            weights = np.frombuffer(before[woff:woff + groups * 9 * 2048], np.uint16)
            table_data = np.ascontiguousarray(before[foff:foff + groups * tables * 256])
            if conv(u16(got), u16(source), Shape(*source.shape), expected.shape[1], expected.shape[2],
                    u16(weights), table_data.ctypes.data, tables):
                raise AssertionError(f"stride-2 op {node['sequence']} rejected")
            got = got.reshape(expected.shape).view(np.float16).astype(np.float64)
            want = expected.astype(np.float64)
            scale = (np.frombuffer(table_data, np.uint32).reshape(groups, tables, 64)[:, 0, :32] & 0xffff)
            scale = scale.astype(np.uint16).view(np.float16).astype(np.float64).reshape(-1)
            mag = magnitude(source.view(np.float16).astype(np.float64),
                            weights.view(np.float16).astype(np.float64), scale, expected.shape)
            ulp = np.spacing(np.abs(want).astype(np.float16)).astype(np.float64)
            tolerance = mag * 2.0 ** -10 + ulp + 2.0 ** -14
            error = np.abs(got - want)
            ratio = float(np.max(error / tolerance))
            worst = max(worst, ratio)
            if ratio > 1:
                index = np.unravel_index(np.argmax(error / tolerance), error.shape)
                raise AssertionError(f"stride-2 op {node['sequence']} mismatch at {index}: "
                                     f"got {got[index]}, want {want[index]}, ratio {ratio}")
            counts["conv"] += 1
        print(f"PASS qnn16 stride2: space-to-depth {counts['s2d']} exact, "
              f"conv {counts['conv']} cases, worst error/tolerance {worst:.3f}")


if __name__ == "__main__":
    main()
