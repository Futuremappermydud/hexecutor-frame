#!/usr/bin/env python3
"""Build/run the shared Frame staging path on Hexagon simulator and oracle-check it."""
import hashlib, json, re, subprocess
from pathlib import Path
import run_xwb1022_frame_probe as oracle

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SDK=ROOT/'.work/hexagon-sdk/6.6.0.0'
FRAME=ROOT/'frame'
SIM='/sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim'
BASE=['docker','run','--rm','--platform','linux/amd64','-v',f'{SDK}:/sdk:ro',
      '-v',f'{FRAME}:/src','-w','/src','depth-model-hmx-tools:latest']

def main():
    subprocess.run([str(HERE/'make_xwb1022_frame_fixture.py')],cwd=ROOT,check=True)
    subprocess.run([str(HERE/'build_xwb1022_frameprobe_sim.sh')],cwd=ROOT,check=True)
    p=subprocess.run(BASE+[SIM,'-mv75','--quiet','native_kernels/xwb1022_frameprobe_sim.elf'],
                     cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=HERE/'xwb1022-frameprobe-sim.log';log.write_text(p.stdout)
    if p.returncode: raise SystemExit(f'hexagon-sim failed ({p.returncode}); see {log}')
    if 'PASS shared simulator staging/call/drain/download/guards' not in p.stdout:
        raise SystemExit(f'shared staging-path proof did not pass; see {log}')
    if p.stdout.count('RAW_OUTPUT_PHYSICAL_HEX_BEGIN')!=1 or p.stdout.count('RAW_OUTPUT_PHYSICAL_HEX_END')!=1:
        raise SystemExit(f'missing/duplicate physical output markers; see {log}')
    body=p.stdout.split('RAW_OUTPUT_PHYSICAL_HEX_BEGIN',1)[1].split('RAW_OUTPUT_PHYSICAL_HEX_END',1)[0]
    output=bytes.fromhex(''.join(body.split()))
    expected=(HERE/'xwb1022_hmx_output.bin').read_bytes()
    cpu=(HERE/'xwb1022_cpu_output.bin').read_bytes()
    if len(output)!=65536 or output!=expected:
        raise SystemExit('shared VTCM-staged simulator output differs from the established HMX physical result')
    comparison=oracle.validate_outputs(expected,cpu,output)
    if not comparison['cpu_0_or_plus1']:
        raise SystemExit(f'CPU logical oracle mismatch: {comparison}')
    inputs,weights,fi,_,_=oracle.parse_fixture((HERE/'xwb1022_frame_fixture.bin').read_bytes())
    line=re.search(r'STAGING fnv32 input=([0-9a-f]{8}) weights=([0-9a-f]{8}) fi=([0-9a-f]{8})',p.stdout)
    if not line: raise SystemExit(f'missing staged hash line; see {log}')
    got=tuple(int(x,16) for x in line.groups())
    want=tuple(oracle.fnv32(x) for x in (inputs,weights,fi))
    if got!=want: raise SystemExit(f'staged operand digest mismatch: got={got}, expected={want}')
    output_path=HERE/'xwb1022_frameprobe_sim_output.bin';output_path.write_bytes(output)
    report={'passed':True,'hardware_accessed':False,'path':'shared run_kernel staging -> generic HMX -> drain -> arena download',
            'simulator_elf_sha256':hashlib.sha256((HERE/'xwb1022_frameprobe_sim.elf').read_bytes()).hexdigest(),
            'log_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),
            'output_sha256':hashlib.sha256(output).hexdigest(),'simulator_physical_exact':True,
            'logical_compare':comparison,'staged_fnv32':list(got),'expected_fnv32':list(want),
            'guards_and_table_tails':'checked inside shared run_kernel'}
    (HERE/'xwb1022_frameprobe_sim_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
