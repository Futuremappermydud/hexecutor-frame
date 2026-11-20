#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check native float32/FP16 casts against local Mercury QD writes."""
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
    with tempfile.TemporaryDirectory(prefix="qnn16-cast-") as tmp:
        so = Path(tmp) / "libqnn16_cast.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        lib = C.CDLL(str(so))
        to_float = lib.qnn16_tensor_f16_to_f32
        to_float.argtypes = [C.POINTER(C.c_float), C.POINTER(Tensor)]
        to_float.restype = C.c_int
        to_half = lib.qnn16_f32_to_f16
        to_half.argtypes = [C.POINTER(C.c_uint16), C.POINTER(C.c_float), C.c_uint32]
        oracle = Oracle(a.context, a.trace, load(a.known))
        counts = {"to_f32": 0, "to_f16": 0}
        for row, node, before, after in oracle.run(
                want=lambda n: n["type_name"].startswith("Cast_fp16_to_fp32@") or
                               n["type_name"].startswith("Cast_fp32_to_fp16.")):
            source_id, target_id = node["inputs"][0], node["outputs"][0]
            expected = oracle.tensor(target_id, after)
            if expected is None: raise AssertionError(f"no QD output for op {node['sequence']}")
            if node["type_name"].startswith("Cast_fp16_to_fp32"):
                src, blocks = descriptor(oracle.tensors[source_id], before)
                got = np.empty(expected.shape, np.float32)
                if to_float(got.ctypes.data_as(C.POINTER(C.c_float)), C.byref(src)) != 0:
                    raise AssertionError(f"cast rejected op {node['sequence']}")
                counts["to_f32"] += 1
            else:
                source = np.ascontiguousarray(oracle.tensor(source_id, before), dtype=np.float32)
                got = np.empty(expected.shape, np.float16)
                to_half(got.ctypes.data_as(C.POINTER(C.c_uint16)),
                        source.ctypes.data_as(C.POINTER(C.c_float)), source.size)
                counts["to_f16"] += 1
            # QAIRT's vector half->float conversion differs by at most one
            # float32 ULP from a standard exact conversion on this trace.
            if not np.allclose(got, expected, rtol=0, atol=1e-7, equal_nan=True):
                raise AssertionError(f"cast mismatch at op {node['sequence']}")
        print(f"PASS qnn16 casts: {counts}")


if __name__ == "__main__":
    main()
