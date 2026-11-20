#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build, run, and independently check the actual QDRX247 CI64/CO64 fixture."""

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

import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

HERE = (PATHS.native / 'kernels/convolution')
ROOT = (PATHS.repo)
FRAME = ROOT / "frame"
SDK = ROOT / ".work/hexagon-sdk/6.6.0.0"
SIM = "/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim"
DOCKER = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v",
          f"{SDK}:/sdk:ro", "-v", f"{FRAME}:/src", "-w", "/src",
          "depth-model-hmx-tools:latest"]
FIXTURE = (PATHS.native / 'kernels/convolution/xwb_ci64_qdrx247_cpu_prefix_fixture.bin')


def fnv64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return h


def main():
    subprocess.run([".venv-cpu/bin/python", "frame/native_kernels/make_xwb_ci64_qdrx247_fixture.py"],
                   cwd=ROOT, check=True)
    subprocess.run(["sh", "frame/native_kernels/build_xwb_ci64_qdrx247_sim.sh"], cwd=ROOT, check=True)
    p = subprocess.run(DOCKER + [SIM, "-mv75", "--quiet",
                                "native_kernels/qnn_hmx_generic_xwb_qdrx247_ci64_sim.elf"],
                       cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = (PATHS.native / 'kernels/convolution/xwb-ci64-qdrx247-sim.log')
    log_path.write_text(p.stdout)
    if p.returncode:
        raise SystemExit(f"hexagon-sim exit {p.returncode}; see {log_path}")
    if "PASS QDRX247 CI64/CO64 physical outputs are CPU-equal or +1" not in p.stdout:
        raise SystemExit(f"simulator proof did not pass; see {log_path}")
    if p.stdout.count("RAW_OUTPUT_BEGIN") != 1 or p.stdout.count("RAW_OUTPUT_END") != 1:
        raise SystemExit(f"missing or duplicate raw output markers; see {log_path}")
    output = bytes.fromhex("".join(p.stdout.split("RAW_OUTPUT_BEGIN", 1)[1]
                                    .split("RAW_OUTPUT_END", 1)[0].split()))
    fixture = FIXTURE.read_bytes()
    h = struct.unpack_from("<32I", fixture)
    xbytes, ybytes, wbytes, fibytes = h[7] * 2048, h[13] * 2048, h[17], h[18]
    expected = fixture[128 + xbytes + h[12] * 2048:
                       128 + xbytes + (h[12] + 32) * 2048]
    if len(output) != 32 * 2048 or len(expected) != len(output):
        raise SystemExit(f"bad output size simulator={len(output)} cpu={len(expected)}")
    deltas = [a - b for a, b in zip(output, expected)]
    counts = {str(delta): deltas.count(delta) for delta in sorted(set(deltas))}
    if any(delta not in (0, 1) for delta in deltas):
        raise SystemExit(f"independent CPU output check failed: {counts}")
    line = re.search(r"staged_fnv64 X=([0-9a-f]{16}) W=([0-9a-f]{16}) FI=([0-9a-f]{16}) Yref=([0-9a-f]{16})", p.stdout)
    if not line:
        raise SystemExit("missing HMX-staged operand hashes")
    got = [int(s, 16) for s in line.groups()]
    x = fixture[128:128 + xbytes]
    y = fixture[128 + xbytes:128 + xbytes + ybytes]
    w = fixture[128 + xbytes + ybytes:128 + xbytes + ybytes + wbytes]
    fi = fixture[128 + xbytes + ybytes + wbytes:128 + xbytes + ybytes + wbytes + fibytes]
    active_y = y[h[12] * 2048:(h[12] + 32) * 2048]
    want = [fnv64(x), fnv64(w), fnv64(fi), fnv64(active_y)]
    if got != want:
        raise SystemExit(f"staged-VTCM hashes differ from exact fixture bytes: {got} != {want}")
    output_path = (PATHS.native / 'kernels/convolution/xwb_ci64_qdrx247_hmx_output.bin')
    output_path.write_bytes(output)
    report = {
        "passed": True,
        "sequence": 247,
        "variant": "ConvLayer_s1.opt@CB*2.Xwb.Fi.t",
        "geometry": {"input_dims": [10, 130, 64], "input_max": [24, 144, 64],
                     "input_padding": [7, 7], "input_pages": 108,
                     "output_dims": [8, 128, 64], "output_max": [8, 128, 64],
                     "output_slice_offset": 100, "output_active_pages": 32,
                     "weights": [3, 3, 64, 64], "weights_format": "Xwb"},
        "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
        "simulator_elf_sha256": hashlib.sha256(((PATHS.native / 'kernels/convolution/qnn_hmx_generic_xwb_qdrx247_ci64_sim.elf')).read_bytes()).hexdigest(),
        "simulator_log_sha256": hashlib.sha256(p.stdout.encode()).hexdigest(),
        "cpu_active_output_sha256": hashlib.sha256(expected).hexdigest(),
        "hmx_output_sha256": hashlib.sha256(output).hexdigest(),
        "hmx_output_is_byte_exact_or_plus_one_to_cpu": True,
        "delta_histogram": counts,
        "staged_fnv64": {"X": f"{got[0]:016x}", "W": f"{got[1]:016x}",
                         "FI": f"{got[2]:016x}", "Y_reference": f"{got[3]:016x}"},
        "guard_and_block_table_tail_checks": "passed inside simulator C",
        "device_accessed": False,
    }
    ((PATHS.native / 'kernels/convolution/xwb_ci64_qdrx247_sim_report.json')).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
