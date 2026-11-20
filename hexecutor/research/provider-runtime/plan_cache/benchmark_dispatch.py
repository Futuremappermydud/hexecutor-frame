#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Host proxy for command decode/handler resolution cost; see README scope."""
from __future__ import annotations

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "depth-model/python"),
                             str(_source_root / "depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()


import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = (PATHS.native / 'program')
sys.path.insert(0, str(ROOT))
import prepared_program as pp  # noqa: E402


def run(path: Path, rounds: int) -> dict:
    raw = path.read_bytes()
    # v2 adds flags to the header but keeps the 20-byte command records and
    # section directory at byte 144. Decode only that immutable section.
    import struct
    magic, version, header_bytes, total_bytes = struct.unpack_from("<4sHHI", raw)
    assert magic == b"NPRG" and version == 2 and header_bytes == 256
    assert total_bytes == len(raw)
    command_count = struct.unpack_from("<I", raw, 32)[0]
    command_off, command_bytes = struct.unpack_from("<II", raw, 180 + 3 * 8)
    assert command_bytes == command_count * pp.COMMAND_BYTES
    command_blob = raw[command_off:command_off + command_bytes]
    commands = [struct.unpack_from("<HBBHHHHHBBBBBB", command_blob,
                                   i * pp.COMMAND_BYTES)
                for i in range(command_count)]
    # Populate the same 80-slot open-addressed hash shape as the C executor,
    # using distinct signatures present in this model. Presence is synthetic:
    # this does not assert that these handlers are supported by the runtime.
    signatures = list(dict.fromkeys((c[1], c[2]) for c in commands))
    table = [None] * 80
    for sig in signatures:
        at = (sig[0] * 37 + sig[1] * 17) % 80
        while table[at] is not None:
            at = (at + 1) % 80
        table[at] = sig
    def resolve(op: int, variant: int) -> int:
        at = (op * 37 + variant * 17) % 80
        for _ in range(80):
            entry = table[at]
            if entry is None:
                raise KeyError((op, variant))
            if entry == (op, variant):
                return at
            at = (at + 1) % 80
        raise KeyError((op, variant))
    tags = [resolve(c[1], c[2]) for c in commands]
    stride = pp.COMMAND_BYTES

    def decode_probe() -> int:
        total = 0
        for i in range(len(commands)):
            # Match the NPRG2 command tuple decode shape, then hash/probe a
            # registry map on every command (the current executor's pattern).
            _, op, variant, *_ = struct.unpack_from(
                "<HBBHHHHHBBBBBB", command_blob, i * stride)
            total += resolve(op, variant)
        return total

    def cached_tags() -> int:
        total = 0
        for tag in tags:
            total += tag
        return total

    def measure(fn):
        samples = []
        expected = None
        for _ in range(rounds):
            start = time.perf_counter_ns()
            result = fn()
            samples.append(time.perf_counter_ns() - start)
            if expected is None:
                expected = result
            assert result == expected
        return {"median_ns": int(statistics.median(samples)),
                "min_ns": min(samples), "max_ns": max(samples)}

    return {
        "image": path.name,
        "commands": len(commands),
        "unique_signatures": len(signatures),
        "tag_table_bytes": (len(commands) + 127) & ~127,
        "rounds": rounds,
        "decode_and_registry_lookup": measure(decode_probe),
        "cached_tag_walk": measure(cached_tags),
        "coverage": "synthetic registry entries; no handler validation executed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--output", type=Path)
    ap.add_argument("images", nargs="*", type=Path,
                    default=[(PATHS.model_deployment / 'programs/primary-v2.nprg'),
                             (PATHS.model_deployment / 'programs/alternate-v2.nprg')])
    args = ap.parse_args()
    report = {"scope": "host dispatch-only proxy; not validation or device latency",
              "results": [run(x, args.rounds) for x in args.images]}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
