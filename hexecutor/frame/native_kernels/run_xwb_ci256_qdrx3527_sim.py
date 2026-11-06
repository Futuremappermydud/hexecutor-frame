#!/usr/bin/env python3
"""Run and independently check actual QDRX3527 CI256/CO128 HMX simulation."""
import hashlib,json,re,struct,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];FRAME=ROOT/'frame';SDK=ROOT/'.work/hexagon-sdk/6.6.0.0'
SIM='/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim'
DOCKER=['docker','run','--rm','--platform','linux/amd64','-v',f'{SDK}:/sdk:ro','-v',f'{FRAME}:/src','-w','/src','depth-model-hmx-tools:latest']
FIXTURE=HERE/'xwb_ci256_qdrx3527_cpu_prefix_fixture.bin'
def fnv64(b):
 h=0xcbf29ce484222325
 for v in b:h=((h^v)*0x100000001b3)&0xffffffffffffffff
 return h
def main():
 subprocess.run(['.venv-cpu/bin/python','frame/native_kernels/make_xwb_ci256_qdrx3527_fixture.py'],cwd=ROOT,check=True)
 subprocess.run(['sh','frame/native_kernels/build_xwb_ci256_qdrx3527_sim.sh'],cwd=ROOT,check=True)
 p=subprocess.run(DOCKER+[SIM,'-mv75','--quiet','native_kernels/qnn_hmx_generic_xwb_qdrx3527_ci256_sim.elf'],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 log=HERE/'xwb-ci256-qdrx3527-sim.log';log.write_text(p.stdout)
 if p.returncode or 'PASS QDRX3527 CI256/CO128 physical output CPU equal or +1' not in p.stdout:raise SystemExit(f'simulator proof failed ({p.returncode}); see {log}')
 if p.stdout.count('RAW_OUTPUT_BEGIN')!=1 or p.stdout.count('RAW_OUTPUT_END')!=1:raise SystemExit('missing output markers')
 out=bytes.fromhex(''.join(p.stdout.split('RAW_OUTPUT_BEGIN',1)[1].split('RAW_OUTPUT_END',1)[0].split()))
 f=FIXTURE.read_bytes();h=struct.unpack_from('<32I',f);xb,yb,wb,fb=h[7]*2048,h[13]*2048,h[17],h[18]
 x=f[128:128+xb];y=f[128+xb:128+xb+yb];w=f[128+xb+yb:128+xb+yb+wb];fi=f[128+xb+yb+wb:128+xb+yb+wb+fb]
 active=((h[10]+7)//8)*((h[11]+7)//8)*(h[15]//32);expected=y[h[12]*2048:(h[12]+active)*2048]
 if len(out)!=len(expected):raise SystemExit(f'output sizes differ HMX={len(out)} CPU={len(expected)}')
 ds=[a-b for a,b in zip(out,expected)];hist={str(d):ds.count(d) for d in sorted(set(ds))}
 if any(d not in (0,1) for d in ds):raise SystemExit(f'CPU output mismatch outside 0/+1: {hist}')
 m=re.search(r'staged_fnv64 X=([0-9a-f]{16}) W=([0-9a-f]{16}) FI=([0-9a-f]{16}) Yref=([0-9a-f]{16})',p.stdout)
 if not m:raise SystemExit('staging hash line missing')
 got=[int(s,16) for s in m.groups()];want=[fnv64(x),fnv64(w),fnv64(fi),fnv64(expected)]
 if got!=want:raise SystemExit(f'staging hashes differ: got {got}, expected {want}')
 outpath=HERE/'xwb_ci256_qdrx3527_hmx_output.bin';outpath.write_bytes(out)
 report={'passed':True,'sequence':3527,'variant':'ConvLayer_s1.opt@CB*2.Xwb.Fi.t','geometry':{'input_dims':[h[2],h[3],h[14]],'input_max':[h[4],h[5],h[14]],'input_padding':[h[19],h[20]],'input_pages':h[7],'output_dims':[h[8],h[9],h[15]],'output_max':[h[10],h[11],h[15]],'output_slice_offset':h[12],'output_active_pages':active,'weight_dims':[3,3,h[14],h[15]],'weight_format':'Xwb'},'fixture_sha256':hashlib.sha256(f).hexdigest(),'simulator_elf_sha256':hashlib.sha256((HERE/'qnn_hmx_generic_xwb_qdrx3527_ci256_sim.elf').read_bytes()).hexdigest(),'simulator_log_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'cpu_active_output_sha256':hashlib.sha256(expected).hexdigest(),'hmx_output_sha256':hashlib.sha256(out).hexdigest(),'hmx_output_is_byte_exact_or_plus_one_to_cpu':True,'delta_histogram':hist,'staged_fnv64':dict(zip(['X','W','FI','Y_reference'],[f'{v:016x}' for v in got])),'guard_and_block_table_tail_checks':'passed inside simulator C','device_accessed':False}
 (HERE/'xwb_ci256_qdrx3527_sim_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
