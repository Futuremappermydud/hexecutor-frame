#!/usr/bin/env python3
"""Turn a successful graph-direct SDK simulation marker into a pinned report."""
import argparse
import hashlib
import json
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / "frame/native_runtime"
EXPECTED = "cfec6dbed3eb2e88544d5c0371c8c50915ce90e6b14f44be7c4411c19bf5e15f"

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("direct", "stackcopy"), required=True)
args = parser.parse_args()
marker = (RUN / "direct-3x3-graph-caller.ok").read_bytes()
if len(marker) != 20:
    raise SystemExit("FAIL graph caller marker size")
magic, seq, abi_mismatches, table_mismatches, stackcopy = struct.unpack("<5I", marker)
if (magic, seq, abi_mismatches, table_mismatches, stackcopy) != (
    0x33444347, 247, 0, 0, int(args.mode == "stackcopy")
):
    raise SystemExit("FAIL graph caller marker contents")
raw = (RUN / "direct-3x3-scatter-output.bin").read_bytes()
if len(raw) != 128 + 32 * 2048:
    raise SystemExit("FAIL graph caller output extent")
output_sha = hashlib.sha256(raw[128:]).hexdigest()
if output_sha != EXPECTED:
    raise SystemExit(f"FAIL graph caller output SHA {output_sha}")
report = {
    "passed": True,
    "offline_only": True,
    "sequence": 247,
    "mode": args.mode,
    "actual_graph_direct_run": True,
    "fixed_l2_tables": "0x30000000",
    "l2_sim_physical_base": "0xd8000000",
    "sim_vtcm_base": "0xd9000000",
    "abi_word_mismatches_vs_leaf_reference": abi_mismatches,
    "active_table_word_mismatches_vs_leaf_reference": table_mismatches,
    "leaf_output_sha256": output_sha,
    "expected_staged_sha256": EXPECTED,
    "source_sha256": {
        name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        for name in ("direct_3x3_scatter_sim.c", "qd_native_direct.c", "qd_native_direct.h")
    },
    "checks_in_target_harness": [
        "graph range preflight and dispatch through qd_graph_native_direct.h",
        "seq247 input/output tables with duplicate refs and independent physical-page permutation",
        "pool1 boundary copy-in/copyback and restored host pool pointer",
        "all active VTCM bytes outside output pages unchanged",
        "zero ABI/table differences from the standalone prepared leaf",
    ],
}
path = RUN / f"direct-3x3-graph-{args.mode}-report.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
