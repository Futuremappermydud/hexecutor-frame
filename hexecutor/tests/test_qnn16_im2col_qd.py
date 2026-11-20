#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Replay Mercury's native FP32-to-FP16 im2col against ignored QD writes."""
import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qnn16_format_qd import Shape, Tensor, descriptor  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from qnn16_oracle import Oracle, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("context", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--known", type=Path, required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-im2col-") as tmp:
        so = Path(tmp) / "libqnn16_im2col.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        native = C.CDLL(str(so)).qnn16_tensor_im2col_cast
        native.argtypes = [C.POINTER(Tensor), C.POINTER(C.c_float), Shape]
        native.restype = C.c_int
        oracle = Oracle(a.context, a.trace, load(a.known))
        count = 0
        for row, node, before, after in oracle.run(
                want=lambda n: "inconv_im2col_cast" in n["type_name"]):
            source = np.ascontiguousarray(oracle.tensor(node["inputs"][0], before), dtype=np.float32)
            target = oracle.tensors[node["outputs"][0]]
            expected = oracle.tensor(target["id"], after)
            memory = before.copy()
            dst, blocks = descriptor(target, memory)
            if native(C.byref(dst), source.ctypes.data_as(C.POINTER(C.c_float)),
                      Shape(*source.shape)) != 0:
                raise AssertionError(f"native im2col rejected op {node['sequence']}")
            got = oracle.tensor(target["id"], memory)
            if not np.array_equal(got, expected):
                raise AssertionError(f"im2col mismatch at op {node['sequence']}")
            count += 1
        print(f"PASS qnn16 im2col cast: {count} traced operations")


if __name__ == "__main__":
    main()
