#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build and run the scalar QNN16 sigmoid adapter against a QD_WRITES sample.

The fixture stays in a temporary directory; no captured model arrays are
written into the repository.
"""
import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("context", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--known", type=Path, required=True)
    a = p.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / "hexecutor/offline"))
    from qnn16_oracle import Oracle, load
    oracle = Oracle(a.context, a.trace, load(a.known))
    sample = None
    for row, node, before, after in oracle.run():
        if node and node["type_name"].startswith("Sigmoid.fp16"):
            src, = node["inputs"]
            dst, = node["outputs"]
            x = oracle.tensor(src, before).reshape(-1).astype(np.float16)
            y = oracle.tensor(dst, after).reshape(-1).astype(np.float16)
            sample = (x, y)
            break
    if sample is None:
        raise SystemExit("trace has no Sigmoid.fp16 operation with decoded tensors")
    x, y = sample
    with tempfile.TemporaryDirectory(prefix="qnn16-sigmoid-") as td:
        td = Path(td)
        fixture = td / "qd.bin"
        fixture.write_bytes(struct.pack("<I", len(x)) + x.tobytes() + y.tobytes())
        exe = td / "test_qd"
        subprocess.run(["cc", "-std=c11", "-O2", "-I", str(ROOT / "hexecutor/native/qnn16"),
                        str(ROOT / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        str(ROOT / "hexecutor/tests/test_qnn16_sigmoid_qd.c"), "-lm", "-o", str(exe)], check=True)
        subprocess.run([str(exe), str(fixture)], check=True)

if __name__ == "__main__":
    main()
