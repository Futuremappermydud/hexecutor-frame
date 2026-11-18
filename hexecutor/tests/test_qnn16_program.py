#!/usr/bin/env python3
"""Run a lowered Q16P program on the host and compare it with QNN.

The portable interpreter (hexecutor/native/qnn16/exec.c) executes the
program with the float 1x1 reference in place of HMX. With --trace, every
operation's output tensor is compared with the QNN simulator's VTCM state
after the same node. The final outputs are compared with the simulator's
Result_0 files and must keep every heatmap argmax.

Usage: python3.14 hexecutor/tests/test_qnn16_program.py PREFIX --run TRACE_DIR --inputs DIR
       [--context CONTEXT --known FUNCTIONS [--kinds A,B]] (enables per-operation comparison)
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hexecutor/tests"))
sys.path.insert(0, str(REPO / "hexecutor/offline"))
from test_qnn16_io_qd import metadata_names, model_inputs  # noqa: E402

VTCM = 8 << 20
KINDS = ["COPY", "ZERO", "CONV1X1", "DW1X1", "DW3X3", "CONV3X3", "S2D", "DW3X3_S2", "IM2COL",
         "INPUT", "OUTPUT", "FORMAT", "CAST_F32", "CAST_F16", "MUL", "MINMAX", "SIGMOID",
         "RESHAPE", "CONCAT", "SLICE", "SHUFFLE", "TRANSPOSE", "AVGPOOL", "X2S", "ZAP", "PW1X1", "SWISH", "WELFORD_PARTIAL", "CONV3X3_S2", "WELFORD_MERGE", "MOVE", "ADD_SWISH", "GROUP_MOMENTS", "DECIMATE", "SLICE_F32", "INSTANCE_NORM", "RESIZE_BILINEAR", "SOFTMAX", "PW_SWISH", "REDUCE_SUM", "RESIZE_NEAREST",
         "L2NORM", "REDUCE_MEAN", "PW_TANH", "AFFINE"]


def build(tmp):
    so = Path(tmp) / "libq16host.so"
    native = REPO / "hexecutor/native/qnn16"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC", f"-I{native}",
                    str(native / "program.c"), str(native / "exec.c"), str(native / "reference.c"),
                    str(native / "qnn16_f32.c"),
                    str(native / "qnn16_hvx.c"),
                    str(REPO / "hexecutor/tests/qnn16_host_run.c"), "-lm", "-o", str(so)], check=True)
    lib = C.CDLL(str(so))
    lib.q16_host_open.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint32)]
    lib.q16_host_bind.argtypes = [C.c_void_p] * 4
    lib.q16_host_step.argtypes = [C.c_uint32]
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", type=Path, help="lowering output prefix (.q16p/.q16c/.q16.json)")
    ap.add_argument("--run", type=Path, required=True, help="simulator run with Result_0 and metadata")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--context", type=Path)
    ap.add_argument("--known", type=Path)
    ap.add_argument("--kinds", help="comma-separated op kinds for the per-operation comparison (default: all)")
    a = ap.parse_args()
    image = np.frombuffer(a.prefix.with_suffix(".q16p").read_bytes(), np.uint8).copy()
    constants = np.frombuffer(a.prefix.with_suffix(".q16c").read_bytes(), np.uint8).copy()
    summary = json.loads(a.prefix.with_suffix(".q16.json").read_text())
    header = np.frombuffer(image[:64].tobytes(), np.uint32)
    ops, inputs, outputs = int(header[4]), int(header[6]), int(header[7])
    scratch_bytes, exchange_bytes = int(header[13]), int(header[14])
    with tempfile.TemporaryDirectory(prefix="qnn16-program-") as tmp:
        lib = build(tmp)
        bad = C.c_uint32()
        err = lib.q16_host_open(image.ctypes.data, image.size, constants.size, C.byref(bad))
        if err:
            raise AssertionError(f"program rejected: error {err} at op {bad.value}")
        vtcm = np.zeros(VTCM, np.uint8)
        exchange = np.zeros(exchange_bytes, np.uint8)
        scratch = np.zeros(scratch_bytes, np.uint8)
        lib.q16_host_bind(vtcm.ctypes.data, exchange.ctypes.data, scratch.ctypes.data, constants.ctypes.data)
        input_names, output_names = metadata_names(a.run)
        values = model_inputs(a.run, a.inputs)
        for slot, (dims, name) in enumerate(zip(summary["inputs"], input_names)):
            off = summary_io_offset(image, header, slot)
            data = values[name].astype(np.float32)
            exchange[off:off + data.nbytes] = np.frombuffer(data.tobytes(), np.uint8)

        oracle = None
        kinds = set(a.kinds.split(",")) if a.kinds else None
        if a.context:
            from qnn16_oracle import Oracle, load
            oracle = Oracle(a.context, a.run, load(a.known))
            states = per_node_states(oracle, set(summary["op_nodes"]))
            # A node whose trace row recorded no changed VTCM block has nothing to compare (for
            # example a one-element input that already held its value in the simulator).
            for row in oracle.rows:
                node = oracle.nodes[(row[1], row[2])]
                if node is not None and row[4] == 0:
                    states.pop(node["sequence"], None)
        worst = {}
        for i in range(ops):
            err = lib.q16_host_step(i)
            if err:
                raise AssertionError(f"op {i} ({KINDS[op_kind(image, header, i) - 1]}) failed: {err}")
            if oracle is not None:
                compare_op(oracle, states, summary, image, header, vtcm, i, worst, kinds)
        if worst:
            print("per-op worst error / tolerance: " +
                  ", ".join(f"{k} {v:.3f}" for k, v in sorted(worst.items())))
        for slot, name in enumerate(output_names):
            off = summary_io_offset(image, header, inputs + slot)
            want = np.fromfile(a.run / "Result_0" / f"{name}.raw", np.float32)
            got = np.frombuffer(exchange[off:off + want.nbytes].tobytes(), np.float32)
            err = np.abs(got - want)
            print(f"{name}: max abs error {err.max():.5f}, max |value| {np.abs(want).max():.3f}")
            if name.startswith("heatmap"):
                dims = summary["outputs"][slot]
                g = got.reshape(dims)
                w = want.reshape(dims)
                gx = g.reshape(-1, g.shape[-1]).argmax(0) if name == "heatmap_xy" else g.reshape(g.shape[-2], -1).argmax(1)
                wx = w.reshape(-1, w.shape[-1]).argmax(0) if name == "heatmap_xy" else w.reshape(w.shape[-2], -1).argmax(1)
                # A different peak is accepted only as a near-tie: QNN's own
                # heatmap must rate our peak within 0.002 of its maximum.
                wf = w.reshape(-1, w.shape[-1]) if name == "heatmap_xy" else w.reshape(w.shape[-2], -1).T
                for j in np.flatnonzero(gx != wx):
                    if wf[wx[j], j] - wf[gx[j], j] > 0.002:
                        raise AssertionError(f"{name}: joint {j} peak {gx[j]} vs {wx[j]}")
                    print(f"{name}: joint {j} near-tie, peaks {gx[j]} and {wx[j]} differ by "
                          f"{wf[wx[j], j] - wf[gx[j], j]:.5f} in QNN")
            if err.max() > 0.05 + 0.02 * np.abs(want).max():
                raise AssertionError(f"{name} differs from QNN")
        print(f"PASS qnn16 program {a.prefix.name}: {ops} ops")


def summary_io_offset(image, header, index):
    io_offset = int(header[11])
    rec = np.frombuffer(image[io_offset + 32 * index:io_offset + 32 * index + 32].tobytes(), np.uint32)
    return int(rec[4])


def op_kind(image, header, i):
    return int(np.frombuffer(image[int(header[9]) + 64 * i:int(header[9]) + 64 * i + 4].tobytes(), np.uint32)[0])


def per_node_states(oracle, wanted):
    """VTCM after each wanted node (by graph sequence)."""
    out = {}
    for _row, node, _before, after in oracle.run(want=lambda n: n["sequence"] in wanted):
        out[node["sequence"]] = after.copy()
    return out


def compare_op(oracle, states, summary, image, header, vtcm, i, worst, kinds=None):
    seq = summary["op_nodes"][i]
    if seq not in states:
        return
    node = next(n for n in oracle.graph["nodes"] if n["sequence"] == seq)
    kind = KINDS[op_kind(image, header, i) - 1]
    if kind in ("OUTPUT", "ZAP", "COPY") or not node["outputs"] or (kinds is not None and kind not in kinds):
        return
    tid = node["outputs"][0]
    want = oracle.tensor(tid, states[seq])
    got = oracle.tensor(tid, vtcm)
    if want is None:
        return
    w, g = want.astype(np.float64), got.astype(np.float64)
    # Compared against the running state, so FP16 drift accumulates op to op.
    tol = 2.0 ** -5 * np.maximum(np.abs(w), 1.0)
    ratio = float(np.max(np.abs(g - w) / tol)) if w.size else 0.0
    worst[kind] = max(worst.get(kind, 0.0), ratio)
    if ratio > 1:
        idx = np.unravel_index(np.argmax(np.abs(g - w) / tol), w.shape)
        raise AssertionError(f"op {i} {kind} (node {seq}) differs at {idx}: got {g[idx]} want {w[idx]}")


if __name__ == "__main__":
    main()
