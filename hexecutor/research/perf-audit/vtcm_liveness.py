#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Offline conservative pool-1 live-block census for a QDRX command schedule.

Usage: python3 frame/perf_audit/vtcm_liveness.py [image.qdrx]
No device access. A tensor is considered live from its first through its last
command reference. Pool-1 @Fill/@Spill extents are additionally marked live for
their command. Output is a planning bound, not a replacement memory planner.
"""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import json
import math
from collections import defaultdict
from pathlib import Path
import sys

ROOT = (PATHS.repo)
sys.path.insert(0, str((PATHS.native)))
from depth_model.lowering.resident_image import unpack

BLOCK_BYTES = 2048


def analyze(path):
    raw = path.read_bytes()
    graph = unpack(raw)
    commands = graph["commands"]
    tensors = {t["id"]: t for t in graph["tensors"]}
    ncmd = len(commands)
    pool_bytes = graph["pool_bytes"]["1"]
    nblocks = pool_bytes // BLOCK_BYTES
    life = {tid: [ncmd, -1] for tid in tensors}
    for index, command in enumerate(commands):
        for tid in set(command["inputs"] + command["outputs"]):
            life[tid][0] = min(life[tid][0], index)
            life[tid][1] = max(life[tid][1], index)
    for tid in graph["graph_inputs"]:
        life[tid][0] = 0
    for tid in graph["graph_outputs"]:
        life[tid][1] = ncmd - 1

    # Per-command events let us scan the 8 MiB pool once per graph command.
    events = defaultdict(list)
    ever = set()

    def add_live_range(lo, hi, start, end):
        if lo > hi or start >= end:
            return
        if start < 0 or end > pool_bytes:
            raise ValueError(f"pool-1 range [{start}, {end}) exceeds {pool_bytes} bytes")
        first = start // BLOCK_BYTES
        last = (end + BLOCK_BYTES - 1) // BLOCK_BYTES
        for block in range(first, last):
            ever.add(block)
            events[lo].append((block, 1))
            events[hi + 1].append((block, -1))

    for tid, tensor in tensors.items():
        lo, hi = life[tid]
        if lo > hi:
            continue
        storage = tensor.get("storage") or {}
        direct = storage.get("direct")
        if direct and direct["pool"] == 1:
            shape = tensor.get("shape")
            size = (math.prod(shape["max_dimensions"]) * tensor["element_bytes"]
                    if shape else 1)
            add_live_range(lo, hi, direct["offset"], direct["offset"] + size)
        else:
            for block in storage.get("blocks", []):
                if block["pool"] == 1:
                    add_live_range(lo, hi, block["offset"], block["offset"] + BLOCK_BYTES)

    # The framework vector can reference pool 1 outside tensor block tables.
    for index, command in enumerate(commands):
        if command["opcode"] != 26:
            continue
        fv = command["framework_vector"]
        for encoded, size in zip(fv[4::2], fv[5::2]):
            if encoded & 0x7ff == 1:
                start = encoded & ~0x7ff
                add_live_range(index, index, start, start + size)

    live = [0] * nblocks
    conv_runs = []
    for index, command in enumerate(commands):
        for block, delta in events.get(index, ()):
            if 0 <= block < nblocks:
                live[block] += delta
        run = largest = 0
        for count in live:
            if count == 0:
                run += 1
                largest = max(largest, run)
            else:
                run = 0
        if command["opcode"] == 6:
            conv_runs.append((index, largest * BLOCK_BYTES))

    static_run = run = 0
    for block in range(nblocks):
        if block in ever:
            run = 0
        else:
            run += 1
            static_run = max(static_run, run)

    # This uses command tensor-reference intervals only. It does not model
    # lifetimes hidden inside asynchronous workers or source-level aliases.
    return {
        "image": str(path),
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "commands": ncmd,
        "pool1_bytes": pool_bytes,
        "block_bytes": BLOCK_BYTES,
        "static_never_referenced_bytes": static_run * BLOCK_BYTES,
        "static_never_referenced_tail": (
            {"offset": (nblocks - static_run) * BLOCK_BYTES,
             "bytes": static_run * BLOCK_BYTES} if static_run else None),
        "highest_referenced_pool1_end": (max(ever) + 1) * BLOCK_BYTES if ever else 0,
        "conv_commands_opcode6": len(conv_runs),
        "min_largest_live_free_run_at_conv": min((size for _, size in conv_runs), default=0),
        "max_largest_live_free_run_at_conv": max((size for _, size in conv_runs), default=0),
        "conv_commands_with_at_least_512k_free": sum(size >= 512 * 1024 for _, size in conv_runs),
        "limitations": (
            "Conservative first-to-last tensor-reference intervals plus pool-1 op26 spans. "
            "Does not prove no hidden aliases/accesses or asynchronous-worker lifetime. "
            "Treat as a scheduling hypothesis; validate all block IDs and worker completion "
            "before allocating scratch in a live range."
        ),
    }


if __name__ == "__main__":
    default = (PATHS.native / 'build/resident-primary.qdrx')
    print(json.dumps(analyze(Path(sys.argv[1]) if len(sys.argv) > 1 else default), indent=2))
