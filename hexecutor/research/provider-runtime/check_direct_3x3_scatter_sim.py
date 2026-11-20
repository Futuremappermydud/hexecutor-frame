#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify seq247 direct HMX against real scattered QDRX pages and reference."""

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

import hashlib
import json
import math
import struct
from pathlib import Path

HERE=(PATHS.repo / 'hexecutor/research/provider-runtime')
FRAME=(PATHS.native)
fixture=(FRAME/'native_kernels/xwb_ci64_qdrx247_cpu_prefix_fixture.bin').read_bytes()
sidecar=((PATHS.repo / 'hexecutor/research/provider-runtime/direct-3x3-qdrx247-offsets.bin')).read_bytes()
direct=((PATHS.repo / 'hexecutor/research/provider-runtime/direct-3x3-scatter-output.bin')).read_bytes()
staged=(FRAME/'native_kernels/qd_native_3x3_adapter_seq247_output.bin').read_bytes()
h=struct.unpack_from('<32I',fixture)
mh,seq,nx,ny,woff,foff,wb,fb=struct.unpack_from('<8I',sidecar)
if mh!=0x324d4353 or seq!=247 or h[1]!=247:raise SystemExit('bad seq247 metadata')
if len(fixture)!=128+(h[7]+h[13])*2048+h[17]+h[18]:raise SystemExit('fixture length mismatch')
if (nx,ny,wb,fb)!=(h[7],h[13],h[17],h[18]):raise SystemExit('sidecar sizes disagree')
xo=struct.unpack_from(f'<{nx}I',sidecar,32)
yo=struct.unpack_from(f'<{ny}I',sidecar,32+4*nx)
inpages=(h[4]//8)*(h[5]//8)*(h[14]//32)
outpages=(h[10]//8)*(h[11]//8)*(h[15]//32)
assert inpages==108 and outpages==32 and h[12]+outpages<=ny
sources=[]
for x in xo[:inpages]:
    if x not in sources:sources.append(x)
for y in yo[h[12]:h[12]+outpages]:
    if y not in sources:sources.append(y)
np=len(sources)
mul=73
while math.gcd(mul,np)!=1:mul+=2
dest={off:((i*mul)%np)*2048 for i,off in enumerate(sources)}
aliases=inpages-len(set(xo[:inpages]))
nonmono=sum(dest[xo[q]]<dest[xo[q-1]] for q in range(1,inpages))
if len(set(dest.values()))!=np or not aliases or not nonmono:raise SystemExit('scatter stress pattern is insufficient')
if direct[:128]!=fixture[:128]:raise SystemExit('output header mismatch')
got=direct[128:]
expected=fixture[128+h[7]*2048+h[12]*2048:128+h[7]*2048+(h[12]+outpages)*2048]
if len(got)!=outpages*2048 or len(staged)!=len(got):raise SystemExit('output length mismatch')
if got!=staged:raise SystemExit('scattered direct output differs from staged native leaf')
delta=[a-b for a,b in zip(got,expected)]
hist={str(k):delta.count(k) for k in sorted(set(delta))}
if any(v not in (0,1) for v in delta):raise SystemExit(f'CPU-prefix delta outside [0,1]: {hist}')
report={
  'passed':True,'offline_only':True,'sequence':247,
  'input_block_refs':nx,'input_pages_consumed':inpages,
  'unique_input_pages':len(set(xo[:inpages])),'repeated_input_aliases':aliases,
  'output_pages_consumed':outpages,'unique_touched_physical_pages':np,
  'independent_physical_page_permutation_multiplier':mul,
  'nonmonotone_input_table_transitions':nonmono,
  'weight_pool1_offset':hex(woff),'fi_pool1_offset':hex(foff),
  'direct_equals_staged':True,'workspace_tensor_staging_bytes':0,
  'cpu_prefix_delta_histogram':hist,
  'direct_output_sha256':hashlib.sha256(got).hexdigest(),
  'staged_output_sha256':hashlib.sha256(staged).hexdigest(),
  'fixture_sha256':hashlib.sha256(fixture).hexdigest(),
  'source_sha256':{
    'qd_native_direct.c':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.c')).read_bytes()).hexdigest(),
    'qd_native_direct.h':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/qd_native_direct.h')).read_bytes()).hexdigest(),
    'scatter_sim.c':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/direct_3x3_scatter_sim.c')).read_bytes()).hexdigest(),
    'offset_generator.py':hashlib.sha256(((PATHS.repo / 'hexecutor/research/provider-runtime/make_direct_3x3_scatter_map.py')).read_bytes()).hexdigest(),
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
  'checks':'real seq247 pool1 offsets; duplicate X refs preserved; unique scattered VTCM pages independently permuted; output compared bytewise to staged leaf and within [0,+1] CPU-prefix tolerance; C harness verifies source unchanged and VTCM outside scattered output unchanged'
}
((PATHS.repo / 'hexecutor/research/provider-runtime/direct_3x3_scatter_sim_report.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
