#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check Mercury's external I/O and housekeeping leaves against a QD trace.

Covers *InputSlicePad and *InputSlice (model inputs to VTCM), *OutputSlice
and *OutputCast_fp16_to_fp32 (VTCM to the simulator's Result_0 outputs),
Mul.fp16, Reshape, zero_to_vtcm and @BlockZap. Slice origins come from the
operations' decoded s4 parameters. Inputs are read from the trace's
input_list.txt, with each file looked up by basename in --inputs.

Usage: python3.14 hexecutor/tests/test_qnn16_io_qd.py CONTEXT TRACE --known FUNCTIONS --inputs DIR
"""
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

TILE = np.arange(1024)
TILE_ROW = TILE // 128
TILE_COL = ((TILE // 64) % 2) * 2 + TILE % 2


def f32p(a):
    return a.ctypes.data_as(C.POINTER(C.c_float))


def metadata_names(trace):
    """Input and output tensor names, in slot order, from execution_metadata.yaml."""
    names = {"input_tensors": [], "output_tensors": []}
    section = None
    for line in (trace / "execution_metadata.yaml").read_text().splitlines():
        key = line.strip().rstrip(":")
        if key in names:
            section = key
        elif section and line.strip().startswith("- tensor_name:"):
            names[section].append(line.split(":", 1)[1].strip())
    return names["input_tensors"], names["output_tensors"]


def model_inputs(trace, inputs_dir):
    files = {}
    for item in (trace / "input_list.txt").read_text().split():
        name, path = item.split(":=", 1)
        files[name] = np.fromfile(inputs_dir / Path(path).name, np.float32)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("context", type=Path)
    p.add_argument("trace", type=Path)
    p.add_argument("--known", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True,
                   help="directory holding the trace's input files")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="qnn16-io-") as tmp:
        so = Path(tmp) / "libqnn16_io.so"
        subprocess.run(["cc", "-std=c11", "-O2", "-shared", "-fPIC",
                        str(repo / "hexecutor/native/qnn16/qnn16_hvx.c"),
                        "-lm", "-o", str(so)], check=True)
        lib = C.CDLL(str(so))
        lib.qnn16_window_f32.argtypes = [C.POINTER(C.c_float), Shape, C.POINTER(C.c_float), Shape,
                                         C.c_int32, C.c_int32]
        lib.qnn16_place_f32.argtypes = [C.POINTER(C.c_float), Shape, C.POINTER(C.c_float), Shape,
                                        C.c_uint32, C.c_uint32]
        lib.qnn16_tensor_f16_to_f32.argtypes = [C.POINTER(C.c_float), C.POINTER(Tensor)]
        lib.qnn16_tensor_mul.argtypes = [C.POINTER(Tensor)] * 3 + [C.c_void_p, C.c_void_p]
        lib.qnn16_gather_f16.argtypes = [C.POINTER(C.c_uint16), C.POINTER(Tensor)]
        lib.qnn16_zap_region.argtypes = [C.c_uint32, C.POINTER(C.c_uint32), C.POINTER(C.c_uint32)]
        lib.qnn16_zap_blocks.argtypes = [C.POINTER(C.POINTER(C.c_uint8)), C.c_uint32, C.c_uint32, C.c_uint32]

        oracle = Oracle(a.context, a.trace, load(a.known))
        T = oracle.tensors
        input_names, output_names = metadata_names(a.trace)
        inputs = model_inputs(a.trace, a.inputs)
        results = {name: np.fromfile(a.trace / "Result_0" / f"{name}.raw", np.float32)
                   for name in output_names}
        assembled = {name: np.full_like(v, np.nan) for name, v in results.items()}
        counts = {}

        def done(kind):
            counts[kind] = counts.get(kind, 0) + 1

        def s4(node, i):
            return T[node["inputs"][i]]["shape_values"]

        for row, node, before, after in oracle.run(want=lambda n: n["type_name"].startswith((
                "*InputSlice", "*OutputSlice", "*OutputCast", "Mul.fp16", "Reshape",
                "zero_to_vtcm", "@BlockZap"))):
            kind = node["type_name"].split("@")[0] if node["type_name"][0] != "@" else "@BlockZap"
            seq = node["sequence"]
            if kind.startswith("*InputSlice"):
                pad, start, out_dims, slot = s4(node, 0), s4(node, 1), s4(node, 2), s4(node, 4 if kind == "*InputSlicePad" else 3)[-1]
                src_dims = s4(node, -1)
                src = np.ascontiguousarray(inputs[input_names[slot]], np.float32)
                got = np.empty(int(np.prod(out_dims)), np.float32)
                if lib.qnn16_window_f32(f32p(got), Shape(*out_dims), f32p(src), Shape(*src_dims),
                                        start[1] - pad[1], start[2] - pad[2]):
                    raise AssertionError(f"{kind} op {seq} rejected")
                want = oracle.tensor(node["outputs"][0], after).reshape(-1)
                if not np.array_equal(got.view(np.uint32), want.view(np.uint32)):
                    raise AssertionError(f"{kind} op {seq} mismatch")
                done(kind)
            elif kind in ("*OutputSlice", "*OutputCast_fp16_to_fp32"):
                src_t = T[node["inputs"][0]]
                full = s4(node, 1)
                origin = s4(node, 3 if kind == "*OutputCast_fp16_to_fp32" else 2)
                slot = s4(node, -1)[-1]
                name = output_names[slot]
                tile = src_t["shape"]["dimensions"]
                if kind == "*OutputSlice":
                    src = np.ascontiguousarray(oracle.tensor(src_t["id"], before), np.float32)
                else:
                    desc, _blocks = descriptor(src_t, before)
                    src = np.empty(tile, np.float32)
                    if lib.qnn16_tensor_f16_to_f32(f32p(src), C.byref(desc)):
                        raise AssertionError(f"{kind} op {seq} rejected")
                out = assembled[name]
                if lib.qnn16_place_f32(f32p(out), Shape(*full), f32p(src), Shape(*tile),
                                       origin[1], origin[2]):
                    raise AssertionError(f"{kind} op {seq} rejected")
                done(kind)
            elif kind == "Mul.fp16":
                memory = before.copy()
                x, y = (descriptor(T[i], memory) for i in node["inputs"][:2])
                out, _blocks = descriptor(T[node["outputs"][0]], memory)
                if lib.qnn16_tensor_mul(C.byref(out), C.byref(x[0]), C.byref(y[0]), None, None):
                    raise AssertionError(f"Mul op {seq} rejected")
                got = oracle.tensor(node["outputs"][0], memory).astype(np.float32)
                want = oracle.tensor(node["outputs"][0], after).astype(np.float32)
                ulp = np.spacing(np.abs(want).astype(np.float16)).astype(np.float32)
                if np.any(np.abs(got - want) > ulp):
                    raise AssertionError(f"Mul op {seq} differs by more than one ULP")
                done(kind)
            elif kind == "Reshape":
                desc, _blocks = descriptor(T[node["inputs"][0]], before)
                got = np.empty(int(np.prod(T[node["inputs"][0]]["shape"]["dimensions"])), np.uint16)
                if lib.qnn16_gather_f16(got.ctypes.data_as(C.POINTER(C.c_uint16)), C.byref(desc)):
                    raise AssertionError(f"Reshape op {seq} rejected")
                want = oracle.tensor(node["outputs"][0], after).reshape(-1).view(np.uint16)
                if not np.array_equal(got, want):
                    raise AssertionError(f"Reshape op {seq}: not a logical NHWC gather")
                done(kind)
            elif kind == "zero_to_vtcm.fp16":
                want = oracle.tensor(node["outputs"][0], after)
                if np.any(want.view(np.uint16)):
                    raise AssertionError(f"zero_to_vtcm op {seq} left non-zero data")
                done(kind)
            else:  # @BlockZap
                memory = before.copy()
                base = memory.ctypes.data
                for table in node["zap_tables"]:
                    rs, cs = C.c_uint32(), C.c_uint32()
                    if lib.qnn16_zap_region(table["code"], C.byref(rs), C.byref(cs)):
                        raise AssertionError(f"@BlockZap op {seq}: unknown code {table['code']:#x}")
                    ptrs = (C.POINTER(C.c_uint8) * len(table["blocks"]))(
                        *(C.cast(base + b, C.POINTER(C.c_uint8)) for b in table["blocks"]))
                    if lib.qnn16_zap_blocks(ptrs, len(table["blocks"]), rs.value, cs.value):
                        raise AssertionError(f"@BlockZap op {seq} rejected")
                for rec in oracle.writes[oracle.writes["seq"] == row[0]]:
                    b = int(rec["block"]) * 2048
                    if not np.array_equal(memory[b:b + 2048], after[b:b + 2048]):
                        raise AssertionError(f"@BlockZap op {seq} differs in block {rec['block']}")
                if not np.array_equal(memory, after):
                    raise AssertionError(f"@BlockZap op {seq} wrote outside its traced blocks")
                done(kind)

        for name, got in assembled.items():
            want = results[name]
            if np.isnan(got).any():
                continue  # not produced by a traced output leaf
            # QAIRT's x86 qf32 half->float path is within ~1 float32 ULP of an
            # exact conversion (more on FP16 subnormals); copies are exact.
            if np.any(np.abs(got - want) > np.abs(want) * 2.0 ** -21 + 2.0 ** -34):
                raise AssertionError(f"output {name} mismatch: max {np.max(np.abs(got - want))}")
            done(f"output {name}")
        print(f"PASS qnn16 I/O: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
