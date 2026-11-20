#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay full Xpe 3x3 convolutions against Mercury QD writes."""
import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from qnn16_oracle import Oracle, direct_offset, load  # noqa: E402


class Shape(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in "bhwc"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("context", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--known", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-conv3x3-") as tmp:
        so = Path(tmp) / "conv3x3.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        native = C.CDLL(str(so)).qnn16_conv3x3_f16
        native.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), Shape,
                           C.c_uint32, C.c_uint32, C.POINTER(C.c_uint16), C.c_void_p, C.c_uint32]
        oracle = Oracle(a.context, a.trace, load(a.known))
        count = 0
        worst = 0.0
        for row, node, before, after in oracle.run(want=lambda n: (
                n["type_name"].startswith("ConvLayer.fp16.s1") and
                oracle.tensors[n["inputs"][3]].get("scalar") == 0 and
                oracle.tensors[n["inputs"][1]]["shape"]["dimensions"][:2] == [3, 3])):
            x, w, fi, _t, _cfg = node["inputs"]
            source = np.ascontiguousarray(oracle.tensor(x, before)).view(np.uint16)
            expected = oracle.tensor(node["outputs"][0], after)
            wd = oracle.tensors[w]["shape"]["dimensions"]
            ng = (wd[3] + 31) // 32
            kg = (wd[2] + 31) // 32
            tables = oracle.tensors[fi]["shape"]["dimensions"][2]
            _, woff = direct_offset(oracle.tensors[w])
            _, foff = direct_offset(oracle.tensors[fi])
            weights = np.frombuffer(before[woff:woff + 9 * ng * kg * 2048], np.uint16)
            table_data = before[foff:foff + ng * tables * 256]
            got = np.empty(expected.size, np.uint16)
            native(got.ctypes.data_as(C.POINTER(C.c_uint16)),
                   source.ctypes.data_as(C.POINTER(C.c_uint16)), Shape(*source.shape),
                   wd[2], wd[3], weights.ctypes.data_as(C.POINTER(C.c_uint16)),
                   table_data.ctypes.data, tables)
            got = got.reshape(expected.shape).view(np.float16).astype(np.float32)
            want = expected.astype(np.float32)
            ulp = np.spacing(np.abs(want).astype(np.float16)).astype(np.float32)
            xf = source.view(np.float16).astype(np.float32)
            wf = weights.view(np.float16)
            magnitude = np.zeros(expected.shape, np.float32)
            for co in range(expected.shape[3]):
                table = np.frombuffer(table_data, np.uint32).reshape(ng, tables, 64)
                scale = (table[co // 32, 0, co % 32] & 0xffff).astype(np.uint16).view(np.float16)
                for ci in range(source.shape[3]):
                    for ky in range(3):
                        for kx in range(3):
                            q = ky * 3 + 2 - kx
                            wi = ((((co // 32) * kg + ci // 32) * 9 + q) * 16 +
                                  (ci % 32) // 2) * 64 + (co % 32) * 2 + ci % 2
                            magnitude[0, :, :, co] += np.abs(
                                xf[0, ky:ky + expected.shape[1], kx:kx + expected.shape[2], ci] *
                                wf[wi].astype(np.float32) * scale.astype(np.float32))
            tolerance = magnitude * 2.0 ** -10 + ulp + 2.0 ** -14
            active = np.zeros(expected.shape, bool)
            for y in range(expected.shape[1]):
                for x in range(expected.shape[2]):
                    active[0, y, x, :] = np.any(xf[0, y:y + 3, x:x + 3, :] != 0)
            tolerance[~active] = np.inf
            error = np.abs(got - want)
            ratio = np.max(error / tolerance)
            worst = max(worst, float(ratio))
            if not np.all(error <= tolerance):
                index = np.unravel_index(np.argmax(error / tolerance), error.shape)
                raise AssertionError(f"3x3 op {node['sequence']} mismatch at {index}: "
                                     f"got {got[index]}, want {want[index]}, ratio {ratio}")
            count += 1
        print(f"PASS qnn16 conv3x3: {count} cases, worst error/tolerance {worst:.3f}")


if __name__ == "__main__":
    main()
