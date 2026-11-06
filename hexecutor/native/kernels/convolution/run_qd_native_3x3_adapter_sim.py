#!/usr/bin/env python3
"""Exercise the bounded 3x3 adapter with actual QDRX CPU-prefix fixtures."""

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

import hashlib,json,re,struct,subprocess
from pathlib import Path
HERE=(PATHS.native / 'kernels/convolution');ROOT=(PATHS.repo)
FIXTURES=[('247','xwb_ci64_qdrx247_cpu_prefix_fixture.bin','make_xwb_ci64_qdrx247_fixture.py'),
          ('874','xwb_ci128_qdrx874_cpu_prefix_fixture.bin','make_xwb_ci128_qdrx874_fixture.py'),
          ('3589','xwb_ci128_qdrx3589_cpu_prefix_fixture.bin','make_xwb_ci128_qdrx3589_fixture.py'),
          ('3527','xwb_ci256_qdrx3527_cpu_prefix_fixture.bin','make_xwb_ci256_qdrx3527_fixture.py'),
          ('5019','xwb_ci32_co2_qdrx5019_cpu_prefix_fixture.bin','make_xwb_ci32_co2_qdrx5019_fixture.py')]
def fnv64(b):
 h=0xcbf29ce484222325
 for v in b:h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
def main():
 rows=[]
 for seq,name,generator in FIXTURES:
  subprocess.run(['.venv-cpu/bin/python',f'frame/native_kernels/{generator}'],cwd=ROOT,check=True)
  fixture=HERE/name;container_path=f'/src/native_kernels/{name}'
  subprocess.run(['sh','frame/native_kernels/build_qnn_hmx_3x3_adapter_fixture_sim.sh',container_path],cwd=ROOT,check=True)
  sdk=ROOT/'.work/hexagon-sdk/6.6.0.0';frame=ROOT/'frame'
  cmd=['docker','run','--rm','--platform','linux/amd64','-v',f'{sdk}:/sdk:ro','-v',f'{frame}:/src','-w','/src','depth-model-hmx-tools:latest','/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim','-mv75','--quiet','native_kernels/qnn_hmx_3x3_adapter_fixture_sim.elf']
  p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  log=(PATHS.native / f'kernels/convolution/3x3-adapter-seq{seq}-sim.log');log.write_text(p.stdout)
  if p.returncode or 'PASS bounded generic 3x3 adapter prepare/execute/drain/guard/scatter' not in p.stdout:raise SystemExit(f'adapter simulator failed for {seq}; see {log}')
  if p.stdout.count('RAW_OUTPUT_BEGIN')!=1 or p.stdout.count('RAW_OUTPUT_END')!=1:raise SystemExit(f'bad output markers for {seq}')
  out=bytes.fromhex(''.join(p.stdout.split('RAW_OUTPUT_BEGIN',1)[1].split('RAW_OUTPUT_END',1)[0].split()))
  f=fixture.read_bytes();h=struct.unpack_from('<32I',f);xb,yb,wb,fb=h[7]*2048,h[13]*2048,h[17],h[18]
  x=f[128:128+xb]
  if seq=='5019':
   y=f[128+xb+yb:128+xb+2*yb];w=f[128+xb+2*yb:128+xb+2*yb+wb];fi=f[128+xb+2*yb+wb:128+xb+2*yb+wb+fb]
  else:
   y=f[128+xb:128+xb+yb];w=f[128+xb+yb:128+xb+yb+wb];fi=f[128+xb+yb+wb:128+xb+yb+wb+fb]
  pages=((h[10]+7)//8)*((h[11]+7)//8)*(h[15]//32);expected=y[h[12]*2048:(h[12]+pages)*2048]
  if len(out)!=len(expected):raise SystemExit(f'sequence {seq}: output byte count mismatch')
  deltas=[a-b for a,b in zip(out,expected)];hist={str(d):deltas.count(d) for d in sorted(set(deltas))}
  if any(d not in (0,1) for d in deltas):raise SystemExit(f'sequence {seq}: host output oracle mismatch {hist}')
  staged=re.search(r'staged_fnv64 X=([0-9a-f]{16}) W=([0-9a-f]{16}) FI=([0-9a-f]{16})',p.stdout)
  used=re.search(r'workspace_used=(\d+)',p.stdout)
  if not staged or not used:raise SystemExit(f'sequence {seq}: missing staging metadata')
  got=[int(s,16) for s in staged.groups()];want=[fnv64(x),fnv64(w),fnv64(fi)]
  if got!=want:raise SystemExit(f'sequence {seq}: staged operand hashes differ {got} != {want}')
  outpath=(PATHS.native / f'kernels/convolution/qd_native_3x3_adapter_seq{seq}_output.bin');outpath.write_bytes(out)
  rows.append({'sequence':int(seq),'passed':True,'input_channels':h[14],'output_channels':h[15],
    'logical_output_channels':h[24] if h[24] else h[15],
    'input_pages':h[7] and ((h[4]+7)//8)*((h[5]+7)//8)*(h[14]//32),
    'output_pages':pages,'output_slice':h[12],'workspace_used':int(used.group(1)),
    'fixture_sha256':hashlib.sha256(f).hexdigest(),'adapter_simulator_elf_sha256':hashlib.sha256(((PATHS.native / 'kernels/convolution/qnn_hmx_3x3_adapter_fixture_sim.elf')).read_bytes()).hexdigest(),
    'log_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'cpu_active_output_sha256':hashlib.sha256(expected).hexdigest(),
    'adapter_output_sha256':hashlib.sha256(out).hexdigest(),'delta_histogram':hist,
    'staged_fnv64':dict(zip(['X','W','FI'],[f'{v:016x}' for v in got])),
    'workspace_boundary_canary':'passed','adapter_guards_table_tails':'passed','device_accessed':False})
 report={'passed':True,'adapter':'qd_native_3x3_adapter prepare/execute/finish','cases':rows}
 ((PATHS.native / 'kernels/convolution/qd_native_3x3_adapter_sim_report.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
