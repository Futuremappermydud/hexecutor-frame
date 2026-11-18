#!/usr/bin/env python3
"""Poison test for the staging blocks qnn16_scratch.py assigns.

Runs a Q16P program twice on the host interpreter. The second pass fills the
staging blocks of every operation that has them (p[3]/p[4] of CONV3X3,
CONV3X3_S2 and PW1X1) with FP16 NaN just before that operation. The host
float path never uses staging blocks, so if every assigned block is dead at
that point the outputs of the two passes are bit-identical; a block that is
read later would carry NaN into the result.

Usage: python3.14 hexecutor/tests/test_qnn16_scratch_poison.py PREFIX --run TRACE_DIR [--runs N]
"""
import argparse
import ctypes as C
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hexecutor/tests"))
sys.path.insert(0, str(REPO / "hexecutor/offline"))
import test_qnn16_program as tp  # noqa: E402
from test_qnn16_io_qd import metadata_names, model_inputs  # noqa: E402

STAGED = {tp.KINDS.index(k) + 1 for k in ("CONV3X3", "CONV3X3_S2", "PW1X1")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", type=Path)
    ap.add_argument("--run", type=Path, required=True, help="simulator run holding the model inputs")
    ap.add_argument("--runs", type=int, default=2, help="consecutive inferences per pass (state wraps between them)")
    a = ap.parse_args()
    image = np.frombuffer(a.prefix.with_suffix(".q16p").read_bytes(), np.uint8).copy()
    constants = np.frombuffer(a.prefix.with_suffix(".q16c").read_bytes(), np.uint8).copy()
    header = np.frombuffer(image[:64].tobytes(), np.uint32)
    nops, inputs, outputs = int(header[4]), int(header[6]), int(header[7])
    ops = np.frombuffer(image, np.uint32, nops * 16, int(header[9])).reshape(nops, 16)
    words = np.frombuffer(image, np.uint32, int(header[5]), int(header[10]))
    io = np.frombuffer(image, np.uint32, (inputs + outputs) * 8, int(header[11])).reshape(-1, 8)
    names, _ = metadata_names(a.run)
    values = model_inputs(a.run, a.run)
    results, poisoned = [], 0
    with tempfile.TemporaryDirectory(prefix="qnn16-poison-") as tmp:
        lib = tp.build(tmp)
        for poison in (False, True):
            bad = C.c_uint32()
            if lib.q16_host_open(image.ctypes.data, image.size, constants.size, C.byref(bad)):
                raise AssertionError(f"program rejected at op {bad.value}")
            vtcm = np.zeros(tp.VTCM, np.uint8)
            exchange = np.zeros(int(header[14]), np.uint8)
            scratch = np.zeros(int(header[13]), np.uint8)
            lib.q16_host_bind(vtcm.ctypes.data, exchange.ctypes.data, scratch.ctypes.data, constants.ctypes.data)
            for slot, name in enumerate(names):
                data = values[name].astype(np.float32)
                off = int(io[slot][4])
                exchange[off:off + data.nbytes] = np.frombuffer(data.tobytes(), np.uint8)
            outs = []
            for _ in range(a.runs):
                for i in range(nops):
                    o = ops[i]
                    if poison and int(o[0]) in STAGED and o[9]:
                        for off in words[int(o[8]):int(o[8]) + int(o[9])]:
                            vtcm[int(off):int(off) + 2048].view(np.uint16)[:] = 0x7e00
                        poisoned += int(o[9])
                    if lib.q16_host_step(i):
                        raise AssertionError(f"op {i} failed")
                for slot in range(outputs):
                    off, n = int(io[inputs + slot][4]), int(io[inputs + slot][5])
                    outs.append(exchange[off:off + n].copy())
            results.append(outs)
    if not poisoned:
        raise AssertionError("the program has no staging blocks")
    for k, (clean, dirty) in enumerate(zip(*results)):
        if not np.array_equal(clean, dirty):
            d = dirty.view(np.float32)
            diff = np.abs(clean.view(np.float32) - d)
            raise AssertionError(f"output {k % outputs} of run {k // outputs} changed under poison "
                                 f"(max difference {np.nanmax(diff) if not np.isnan(diff).all() else 'nan'}, "
                                 f"{int(np.isnan(d).sum())} NaN)")
    print(f"PASS qnn16 scratch poison: {poisoned} block uses poisoned over {a.runs} runs, outputs bit-identical")


if __name__ == "__main__":
    main()
