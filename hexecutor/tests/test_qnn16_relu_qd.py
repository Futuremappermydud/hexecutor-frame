#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check native FP16 ReLU min/max against local Mercury QD writes."""
import argparse
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_qnn16_format_qd import Tensor, descriptor  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
from qnn16_oracle import Oracle, load  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("context", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--known", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-relu-") as tmp:
        so = Path(tmp) / "libqnn16_relu.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        native = C.CDLL(str(so)).qnn16_tensor_minmax
        native.argtypes = [C.POINTER(Tensor), C.POINTER(Tensor), C.c_void_p,
                           C.c_float, C.c_float]
        native.restype = C.c_int
        oracle = Oracle(a.context, a.trace, load(a.known))
        count = 0
        for row, node, before, after in oracle.run(
                want=lambda n: n["type_name"].startswith("ReluMinMax.fp16")):
            source = oracle.tensors[node["inputs"][0]]
            target = oracle.tensors[node["outputs"][0]]
            expected = oracle.tensor(target["id"], after)
            memory = before.copy()
            src, src_blocks = descriptor(source, memory)
            dst, dst_blocks = descriptor(target, memory)
            if native(C.byref(dst), C.byref(src), None, 0.0, 6.0) != 0:
                raise AssertionError(f"native ReLU rejected op {node['sequence']}")
            got = oracle.tensor(target["id"], memory)
            if not np.array_equal(got, expected):
                raise AssertionError(f"ReLU mismatch at op {node['sequence']}")
            count += 1
        print(f"PASS qnn16 ReLU min/max: {count} traced operations")


if __name__ == "__main__":
    main()
