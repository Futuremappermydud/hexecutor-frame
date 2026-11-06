#!/usr/bin/env python3
"""Execute the predeclared 104-job development screen, preserving all evidence."""
import argparse
import os
import platform
from pathlib import Path
import shutil
import subprocess
from stereo_composite_common import ROOT,digest,read_json,write_json
from stereo_guarded_common import SOURCES,make_plan,verify_complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',default='g++')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    binary,inputs,output=args.binary.resolve(),args.inputs.resolve(),args.output.resolve()
    plan=make_plan(binary,inputs,output)
    if output.exists():
        if not args.resume or read_json(output/'plan.json') != plan:
            raise ValueError('Existing output; resume requires identical frozen source/binary/input/commands')
    else:
        compiler=subprocess.check_output([args.compiler,'--version'],text=True)
        output.mkdir(parents=True)
        write_json(output/'plan.json',plan)
        write_json(output/'environment.json',dict(compiler=compiler,platform=platform.platform(),omp_num_threads=8))
        shutil.copyfile(binary,output/'matcher-binary')
        for s in SOURCES:
            target=output/'sources'/s
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/s).read_bytes())
    if args.plan_only:
        print(f"Frozen {len(plan['jobs'])} development jobs");return
    for index,job in enumerate(plan['jobs'],1):
        folder=output/job['name']
        if (folder/'complete.json').exists() and args.resume:
            verify_complete(folder,job);continue
        if folder.exists() or folder.with_suffix('.log').exists():
            raise ValueError(f'Refusing partial-job overwrite: {folder}')
        folder.parent.mkdir(parents=True,exist_ok=True)
        print(f"[{index}/{len(plan['jobs'])}] {job['name']}",flush=True)
        with folder.with_suffix('.log').open('x') as log:
            subprocess.run(job['command'],env=dict(os.environ,OMP_NUM_THREADS='8'),stdout=log,
                           stderr=subprocess.STDOUT,check=True)
        write_json(folder/'complete.json',dict(job=job,output_sha256={str(f.relative_to(folder)):digest(f)
                   for f in sorted(folder.rglob('*')) if f.is_file()}))
    print('Complete. Analyze the verified development screen.',flush=True)


if __name__=='__main__':main()
