#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay small QNN16 data moves against ignored Mercury QD traces."""
import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from qnn16_oracle import Oracle, load  # noqa: E402


class Shape(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in "bhwc"]


def ptr(array):
    return array.ctypes.data_as(C.POINTER(C.c_uint16))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("context", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--known", type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-moves-") as tmp:
        so = Path(tmp) / "moves.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        lib = C.CDLL(str(so))
        oracle = Oracle(a.context, a.trace, load(a.known))
        kinds = {"ChannelShuffle": 0, "AlignedSlice": 0, "Concat": 0,
                 "x2s_opt": 0, "Transpose.2D": 0, "AvgPool_with_pad": 0}

        def selected(n):
            return any(n["type_name"].startswith(k) for k in kinds)

        for row, node, before, after in oracle.run(want=selected):
            kind = next(k for k in kinds if node["type_name"].startswith(k))
            target = node["outputs"][0]
            expected = oracle.tensor(target, after)
            if expected is None:
                raise AssertionError(f"no QD output for op {node['sequence']}")
            dest = np.zeros(expected.size, np.uint16)
            first = node["inputs"][0]
            if kind == "Concat":
                axis = oracle.tensors[first]["scalar"]
                ids = node["inputs"][1:]
                sources = [np.ascontiguousarray(oracle.tensor(tid, before)).view(np.uint16)
                           for tid in ids]
                shapes = (Shape * len(sources))(*(Shape(*oracle.tensors[tid]["shape"]["dimensions"])
                                                   for tid in ids))
                pointers = (C.POINTER(C.c_uint16) * len(sources))(*(ptr(x) for x in sources))
                lib.qnn16_concat_f16(ptr(dest), pointers, shapes, len(sources), axis)
            else:
                source = np.ascontiguousarray(oracle.tensor(first, before)).view(np.uint16)
                shape = Shape(*source.shape)
                if kind == "ChannelShuffle":
                    groups = oracle.tensors[node["inputs"][1]]["scalar"]
                    lib.qnn16_channel_shuffle_f16(ptr(dest), ptr(source), shape, groups)
                elif kind == "AlignedSlice":
                    start = oracle.tensors[node["inputs"][1]]["shape_values"][3]
                    lib.qnn16_slice_f16(ptr(dest), ptr(source), shape, 3, start, expected.shape[3])
                elif kind == "x2s_opt":
                    lib.qnn16_x2s_f16(ptr(dest), ptr(source), shape)
                elif kind == "AvgPool_with_pad":
                    kernel = oracle.tensors[node["inputs"][1]]["shape_values"][1]
                    lib.qnn16_avgpool_valid_f16(ptr(dest), ptr(source), shape, kernel, kernel)
                else:
                    perm = (C.c_uint32 * 4)(*( (0, 2, 1, 3) if source.shape[1] == 32
                                              else (0, 1, 3, 2)))
                    lib.qnn16_transpose_f16(ptr(dest), ptr(source), shape, perm)
            if not np.array_equal(dest, np.ascontiguousarray(expected).view(np.uint16).reshape(-1)):
                raise AssertionError(f"{kind} mismatch at op {node['sequence']}")
            kinds[kind] += 1
        print("PASS qnn16 moves: " + ", ".join(f"{k}={v}" for k, v in kinds.items()))


if __name__ == "__main__":
    main()
