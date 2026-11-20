#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Simulator proof for removing only the output slice modulo CO-block check."""

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

import hashlib,json,re,struct,subprocess,os
from pathlib import Path

HERE=(PATHS.native / 'kernels/convolution');ROOT=(PATHS.repo)
NAME="xwb_ci128_qdrx3627_slice_cpu_prefix_fixture.bin"
def fnv64(b):
 h=0xcbf29ce484222325
 for v in b:h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h

def main():
 subprocess.run([".venv-cpu/bin/python",f"frame/native_kernels/make_xwb_ci128_qdrx3627_slice_fixture.py"],cwd=ROOT,check=True)
 fixture=HERE/NAME;f=fixture.read_bytes();h=struct.unpack_from("<32I",f)
 if h[1]!=3627 or h[12]%4==0 or h[6]%4:
  raise SystemExit(f"fixture is not intended odd-slice case: seq={h[1]} xslice={h[6]} yslice={h[12]}")
 container_path=f"/src/native_kernels/{NAME}"
 subprocess.run(["sh","frame/native_kernels/build_qnn_hmx_3x3_slice_candidate_sim.sh",container_path],cwd=ROOT,check=True)
 sdk=ROOT/".work/hexagon-sdk/6.6.0.0";frame=ROOT/"frame"
 cmd=["docker","run","--rm","--platform","linux/amd64","-v",f"{sdk}:/sdk:ro","-v",f"{frame}:/src","-w","/src","depth-model-hmx-tools:latest","/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim","-mv75","--quiet","native_kernels/qnn_hmx_3x3_slice_candidate_sim.elf"]
 p=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 log=(PATHS.native / 'kernels/convolution/3x3-slice-candidate-seq3627-sim.log');log.write_text(p.stdout)
 if p.returncode or "PASS candidate generic 3x3 odd-slice prepare/execute/drain/guard/scatter" not in p.stdout:
  raise SystemExit(f"candidate simulator failed; see {log}")
 if p.stdout.count("RAW_OUTPUT_BEGIN")!=1 or p.stdout.count("RAW_OUTPUT_END")!=1:
  raise SystemExit("candidate output markers missing")
 out=bytes.fromhex("".join(p.stdout.split("RAW_OUTPUT_BEGIN",1)[1].split("RAW_OUTPUT_END",1)[0].split()))
 xb,yb,wb,fb=h[7]*2048,h[13]*2048,h[17],h[18]
 yoff=128+xb;woff=yoff+yb;fioff=woff+wb
 y=f[yoff:yoff+yb];w=f[woff:woff+wb];fi=f[fioff:fioff+fb];x=f[128:128+xb]
 pages=(h[10]//8)*(h[11]//8)*(h[15]//32);expected=y[h[12]*2048:(h[12]+pages)*2048]
 if len(out)!=len(expected):raise SystemExit(f"output size mismatch {len(out)} != {len(expected)}")
 delta=[a-b for a,b in zip(out,expected)];hist={str(d):delta.count(d) for d in sorted(set(delta))}
 if any(d not in (0,1) for d in delta):raise SystemExit(f"CPU physical oracle failed: {hist}")
 staged=re.search(r"staged_fnv64 X=([0-9a-f]{16}) W=([0-9a-f]{16}) FI=([0-9a-f]{16})",p.stdout)
 got=[int(z,16) for z in staged.groups()] if staged else []
 want=[fnv64(x),fnv64(w),fnv64(fi)]
 if got!=want:raise SystemExit(f"staged operands differ: {got} != {want}")
 outpath=(PATHS.native / 'kernels/convolution/qd_native_3x3_slice_candidate_seq3627_output.bin');outpath.write_bytes(out)
 used_core=ROOT/'frame'/os.environ.get('QD_SLICE_CORE','native_kernels/qd_native_3x3_slice_candidate.c')
 report={"passed":True,"candidate":"selected bounded adapter source with relaxed output slice eligibility",
  "used_core":str(used_core.relative_to(ROOT)),"used_core_sha256":hashlib.sha256(used_core.read_bytes()).hexdigest(),
  "native_3x3_slices":int(os.environ.get('QDS_NATIVE_3X3_SLICES','0')),
  "frozen_core_c_sha256":hashlib.sha256(((PATHS.native / 'kernels/convolution/qd_native_3x3_adapter.c')).read_bytes()).hexdigest(),
  "candidate_core_c_sha256":hashlib.sha256(((PATHS.native / 'kernels/convolution/qd_native_3x3_slice_candidate.c')).read_bytes()).hexdigest(),
  "fixture_sha256":hashlib.sha256(f).hexdigest(),"sequence":h[1],"input_channels":h[14],"output_channels":h[15],
  "input_slice":h[6],"output_slice":h[12],"input_slice_mod_CI_blocks":h[6]%(h[14]//32),
  "output_slice_mod_CO_blocks":h[12]%(h[15]//32),"input_pages":(h[4]//8)*(h[5]//8)*(h[14]//32),
  "output_pages":pages,"physical_output_bytes":len(out),"delta_histogram":hist,
  "staged_fnv64":dict(zip(("X","W","FI"),[f"{v:016x}" for v in got])),
  "cpu_output_sha256":hashlib.sha256(expected).hexdigest(),"candidate_output_sha256":hashlib.sha256(out).hexdigest(),
  "log_sha256":hashlib.sha256(p.stdout.encode()).hexdigest(),"guard_and_table_tails":"passed",
  "workspace_boundary_canary":"passed","hardware_accessed":False}
 ((PATHS.native / 'kernels/convolution/qd_native_3x3_slice_candidate_seq3627_report.json')).write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2))

if __name__=="__main__":main()
