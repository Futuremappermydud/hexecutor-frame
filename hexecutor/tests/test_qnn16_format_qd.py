#!/usr/bin/env python3
"""Compare native ForceFormat moves with local QAIRT QD_WRITES traces.

Only the temporary compiled helper and ignored local traces contain model data.
Usage: python3.14 hexecutor/tests/test_qnn16_format_qd.py CONTEXT TRACE --known FUNCTIONS
"""
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


class Tensor(C.Structure):
    _fields_ = [("base", C.POINTER(C.c_uint8)),
                ("block_offsets", C.POINTER(C.c_uint32)),
                ("block_count", C.c_uint32), ("base_bytes", C.c_uint32),
                ("direct_offset", C.c_uint32), ("dims", Shape),
                ("full", Shape), ("padding", Shape), ("direct", C.c_uint32)]


def descriptor(t, memory):
    shape = t["shape"]
    dims = Shape(*shape["dimensions"])
    padding = Shape(*shape["padding"])
    full = Shape(*shape["max_dimensions"])
    storage = t.get("storage") or {}
    direct = storage.get("direct")
    blocks = None
    if not direct:
        count = full.b * ((full.h + 7) // 8) * ((full.w + 3) // 4) * ((full.c + 31) // 32)
        start = storage.get("slice_offset", 0)
        entries = storage.get("blocks") or []
        offsets = [entries[i]["offset"] if i < len(entries) and entries[i] and
                   entries[i]["pool"] == 1 else 0xffffffff for i in range(start, start + count)]
        blocks = (C.c_uint32 * count)(*offsets)
    obj = Tensor(memory.ctypes.data_as(C.POINTER(C.c_uint8)), blocks,
                 len(blocks) if blocks is not None else 0, len(memory),
                 direct["offset"] if direct else 0, dims, full, padding, bool(direct))
    return obj, blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("context", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--known", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-format-") as tmp:
        library = Path(tmp) / "libqnn16_format.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(library)], check=True)
        native = C.CDLL(str(library)).qnn16_tensor_copy_format
        native.argtypes = [C.POINTER(Tensor), C.POINTER(Tensor)]
        native.restype = C.c_int
        oracle = Oracle(args.context, args.trace, load(args.known))
        cases = 0
        for row, node, before, after in oracle.run(
                want=lambda n: n["type_name"].startswith("ForceFormat_")):
            source = oracle.tensors[node["inputs"][0]]
            target = oracle.tensors[node["outputs"][0]]
            expected = oracle.tensor(target["id"], after)
            if expected is None:
                raise AssertionError(f"no QD output for operation {node['sequence']}")
            memory = before.copy()
            src, src_blocks = descriptor(source, memory)
            dst, dst_blocks = descriptor(target, memory)
            if native(C.byref(dst), C.byref(src)) != 0:
                raise AssertionError(f"native format rejected operation {node['sequence']}")
            got = oracle.tensor(target["id"], memory)
            if not np.array_equal(got, expected):
                raise AssertionError(f"format mismatch at operation {node['sequence']}")
            cases += 1
        print(f"PASS qnn16 ForceFormat: {cases} traced operations")


if __name__ == "__main__":
    main()
