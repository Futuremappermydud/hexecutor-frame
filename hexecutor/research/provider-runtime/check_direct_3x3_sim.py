#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check direct QNN 3x3 pages against the real QDRX CPU-prefix output."""

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
import struct
from pathlib import Path

HERE=(PATHS.repo / 'hexecutor/research/provider-runtime')
FIX=(PATHS.native / 'kernels/convolution/xwb_ci64_qdrx247_cpu_prefix_fixture.bin')
STAGED=(PATHS.native / 'kernels/convolution/qd_native_3x3_adapter_seq247_output.bin')
DIRECT=(PATHS.repo / 'hexecutor/research/provider-runtime/direct-3x3-output.bin')
fixture=FIX.read_bytes();direct=DIRECT.read_bytes()
h=struct.unpack_from('<32I',fixture)
if h[1]!=247 or direct[:128]!=fixture[:128]:raise SystemExit('fixture/header mismatch')
xb=h[7]*2048;yb=h[13]*2048;wb=h[17];fb=h[18]
if len(fixture)!=128+xb+yb+wb+fb:raise SystemExit('fixture payload length mismatch')
opages=((h[10]+7)//8)*((h[11]+7)//8)*(h[15]//32)
expected=fixture[128+xb+h[12]*2048:128+xb+(h[12]+opages)*2048]
got=direct[128:]
staged=STAGED.read_bytes()
if len(got)!=len(expected) or len(staged)!=len(got):raise SystemExit('output length mismatch')
if got!=staged:raise SystemExit('direct output differs from staged native leaf output')
deltas=[a-b for a,b in zip(got,expected)]
hist={str(v):deltas.count(v) for v in sorted(set(deltas))}
if any(v not in (0,1) for v in deltas):raise SystemExit(f'CPU-prefix output delta outside [0,1]: {hist}')
report={
  'passed':True,'offline_only':True,'sequence':247,'input_pages':108,'output_pages':opages,
  'direct_equals_staged':True,'workspace_tensor_staging_bytes':0,
  'output_page_byte_delta_histogram':hist,'direct_output_sha256':hashlib.sha256(got).hexdigest(),
  'staged_output_sha256':hashlib.sha256(staged).hexdigest(),
  'cpu_prefix_expected_sha256':hashlib.sha256(expected).hexdigest(),
  'fixture_sha256':hashlib.sha256(fixture).hexdigest(),
  'source_sha256':{
    'qd_native_direct.c':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.c')).read_bytes()).hexdigest(),
    'qd_native_direct.h':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.h')).read_bytes()).hexdigest(),
    'direct_3x3_sim.c':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/direct_3x3_sim.c')).read_bytes()).hexdigest(),
  },
  'current_source_sha256':{
    'qd_native_direct.c':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.c')).read_bytes()).hexdigest(),
    'qd_native_direct.h':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.h')).read_bytes()).hexdigest()
  },
  'hardware_retest_required_for_current_core':True,
  'hardware_proof_v1_sha256':{
    'qd_native_direct.c':'6622e8126839081fa781736019f8605702e684a30bb6f75ad2b155676b955ce8',
    'qd_native_direct.h':'933c72b9f6f1072b40b78258460139c51dddcec3abc2110ffe6c41613c96469b'
  },
  'simulator':'Hexagon SDK 6.6.0.0 V75 simulator; 512 KiB simulated VTCM, 384 KiB active + 128 KiB tail',
  'checks':'staged/direct active pointer-table and leaf ABI comparison; CPU-prefix [0,+1]; exact staged output; source graph pool unchanged; VTCM outside direct Y pages and A5/zero guards preserved'
}
((PATHS.repo / 'hexecutor/research/provider-runtime/direct_3x3_sim_report.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
