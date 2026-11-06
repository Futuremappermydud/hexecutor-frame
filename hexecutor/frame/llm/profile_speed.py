#!/usr/bin/env python3
"""Time a single NPU token without a CPU or GPU inference fallback."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from npu_qwen import QwenNPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    engine = QwenNPU()
    counts = defaultdict(int)
    seconds = defaultdict(float)
    matrix_seconds = defaultdict(float)
    command = engine.command
    matvec = engine.matvec

    def timed_command(op, *rest, **kwargs):
        start = time.monotonic()
        try:
            return command(op, *rest, **kwargs)
        finally:
            counts[op] += 1
            seconds[op] += time.monotonic() - start

    def timed_matvec(name, vector):
        start = time.monotonic()
        try:
            return matvec(name, vector)
        finally:
            matrix_seconds[name] += time.monotonic() - start

    engine.command = timed_command
    engine.matvec = timed_matvec
    started = time.monotonic()
    with engine:
        opened = time.monotonic()
        logits = engine.run_token(args.token_id)
        computed = time.monotonic()
        transport_times = {
            "prepare_seconds": engine.transport.prepare_seconds,
            "wait_seconds": engine.transport.wait_seconds,
            "collect_seconds": engine.transport.collect_seconds,
        }
    closed = time.monotonic()
    report = {
        "token_id": args.token_id,
        "next_token_id": int(np.argmax(logits)),
        "total_seconds": closed - started,
        "open_seconds": opened - started,
        "compute_seconds": computed - opened,
        "close_seconds": closed - computed,
        "operator_counts": dict(counts),
        "operator_seconds": dict(seconds),
        "matrix_seconds_top": sorted(matrix_seconds.items(), key=lambda item: -item[1])[:15],
        "npu_call_seconds": engine.call_seconds,
        **transport_times,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
