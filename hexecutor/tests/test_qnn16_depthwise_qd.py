#!/usr/bin/env python3
"""Replay QNN16 depthwise t=514/t=544 operations against Mercury QD writes."""
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
    with tempfile.TemporaryDirectory(prefix="qnn16-depthwise-") as tmp:
        so = Path(tmp) / "depthwise.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        lib = C.CDLL(str(so))
        native1 = lib.qnn16_depthwise1x1_f16
        native1.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), Shape,
                            C.c_uint32, C.POINTER(C.c_uint16), C.c_void_p, C.c_uint32]
        native3 = lib.qnn16_depthwise3x3_f16
        native3.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_uint16), Shape,
                            C.POINTER(C.c_uint16), C.c_void_p, C.c_uint32]
        oracle = Oracle(a.context, a.trace, load(a.known))
        counts = {514: 0, 544: 0}
        worst = 0.0
        for row, node, before, after in oracle.run(want=lambda n: (
                n["type_name"].startswith("ConvLayer.fp16.s1") and
                oracle.tensors[n["inputs"][3]].get("scalar") in counts)):
            x, w, fi, t, _cfg = node["inputs"]
            kind = oracle.tensors[t]["scalar"]
            source = np.ascontiguousarray(oracle.tensor(x, before)).view(np.uint16)
            expected = oracle.tensor(node["outputs"][0], after)
            groups = (expected.shape[3] + 31) // 32
            tables = oracle.tensors[fi]["shape"]["dimensions"][2]
            _, woff = direct_offset(oracle.tensors[w])
            _, foff = direct_offset(oracle.tensors[fi])
            kernel = 3 if kind == 514 else 1
            weights = np.frombuffer(before[woff:woff + groups * (9 * 128 if kind == 514 else 2048)], np.uint16)
            table_data = before[foff:foff + groups * tables * 256]
            got = np.empty(expected.size, np.uint16)
            args = (got.ctypes.data_as(C.POINTER(C.c_uint16)),
                    source.ctypes.data_as(C.POINTER(C.c_uint16)), Shape(*source.shape))
            if kind == 514:
                native3(*args, weights.ctypes.data_as(C.POINTER(C.c_uint16)),
                        table_data.ctypes.data, tables)
            else:
                native1(*args, expected.shape[3], weights.ctypes.data_as(C.POINTER(C.c_uint16)),
                        table_data.ctypes.data, tables)
            got = got.reshape(expected.shape).view(np.float16).astype(np.float32)
            want = expected.astype(np.float32)
            ulp = np.spacing(np.abs(want).astype(np.float16)).astype(np.float32)
            xf = source.view(np.float16).astype(np.float32)
            magnitude = np.zeros(expected.shape, np.float32)
            if kind == 514:
                for co in range(expected.shape[3]):
                    base = (co // 32) * 9 * 64 + ((co % 32) // 2) * 4 + (co % 2) * 3
                    for ky in range(3):
                        for kx in range(3):
                            magnitude[0, :, :, co] += np.abs(
                                xf[0, ky:ky + expected.shape[1], kx:kx + expected.shape[2], co] *
                                weights[base + (ky * 3 + 2 - kx) * 64].view(np.float16).astype(np.float32))
            else:
                for co in range(expected.shape[3]):
                    for j in range(2):
                        magnitude[..., co] += np.abs(xf[..., co + j * 32].astype(np.float32) *
                            weights[(co // 32) * 1024 + ((co % 32) // 2) * 4 + j * 3].view(np.float16).astype(np.float32))
            tolerance = magnitude * 2.0 ** -10 + ulp + 2.0 ** -14
            active = np.zeros(expected.shape, bool)
            for y in range(expected.shape[1]):
                for x in range(expected.shape[2]):
                    if kind == 514:
                        active[0, y, x, :] = np.any(xf[0, y:y + 3, x:x + 3, :] != 0, axis=(0, 1))
                    else:
                        active[0, y, x, :] = np.any(xf[0, y, x, :] != 0)
            tolerance[~active] = np.inf
            error = np.abs(got - want)
            ratio = np.max(error / tolerance)
            worst = max(worst, float(ratio))
            if not np.all(error <= tolerance):
                index = np.unravel_index(np.argmax(error / tolerance), error.shape)
                raise AssertionError(f"t={kind} op {node['sequence']} mismatch at {index}: "
                                     f"got {got[index]}, want {want[index]}, ratio {ratio}")
            counts[kind] += 1
        print(f"PASS qnn16 depthwise: {counts}, worst error/tolerance {worst:.3f}")


if __name__ == "__main__":
    main()
