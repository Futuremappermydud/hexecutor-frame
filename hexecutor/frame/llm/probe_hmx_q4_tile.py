#!/usr/bin/env python3
"""Compare one compressed HMX Q4 tile matrix against an offline CPU reference."""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np

from npu_qwen import QwenNPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", default="blk.0.ffn_up.weight")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    engine = QwenNPU()
    tensor = engine.tensors[args.tensor]
    k, n = map(int, tensor.shape)
    x = np.random.default_rng(2026).normal(0, 0.1, k).astype(np.float32)
    weights = engine.gguf.dequantize(tensor.data, tensor.tensor_type)
    expected = x.astype(np.float16).astype(np.float32) @ weights.T
    started = time.monotonic()
    with engine:
        actual = engine.matvec(args.tensor, x)
        status = list(struct.unpack_from("<32I", engine.transport.arena, 2048))
        debug = [list(struct.unpack_from("<16H", engine.transport.arena, offset))
                 for offset in (0x3000, 0x3080, 0x3100, 0x3180)]
    diff = np.abs(actual - expected)
    report = {"tensor": args.tensor, "k": k, "n": n,
              "seconds": time.monotonic() - started, "npu_calls": engine.calls,
              "npu_call_seconds": engine.call_seconds,
              "max_abs_error": float(np.max(diff)),
              "max_rel_error": float(np.max(diff / np.maximum(0.1, np.abs(expected)))),
              "sample_actual": actual[:16].tolist(), "sample_expected": expected[:16].tolist(),
              "status": status}
    report["debug"] = debug
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["max_rel_error"] > 0.03:
        raise RuntimeError("compressed HMX matrix disagrees with reference")


if __name__ == "__main__":
    main()
