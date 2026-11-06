#!/usr/bin/env python3
"""Root development commands, with explicit package and resource locations."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ)
ENV['PYTHONPATH']=os.pathsep.join(map(str,(ROOT,ROOT/'depth-model/python',ROOT/'depth-model/tools',
                                         ROOT/'hexecutor/benchmarks',ROOT/'hexecutor/native/program')))+os.pathsep+ENV.get('PYTHONPATH','')
def run(*args,cwd=ROOT):subprocess.run(list(map(str,args)),cwd=cwd,env=ENV,check=True)
def source_files():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    return {ROOT/p for p in paths if p and (ROOT/p).is_file()}
def check_docs():
    failures=[];count=0
    for p in (ROOT/'docs').rglob('*.md'):
        for label,url in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',p.read_text()):
            if re.match(r'^[a-z]+:',url):continue
            name=unquote(url.split('#')[0]).strip('<>')
            if name and not (p.parent/name).exists():failures.append((str(p.relative_to(ROOT)),name))
            count+=1
    if failures:raise RuntimeError(f'Broken documentation links: {failures}')
    print(f'Documentation links checked: {count}',flush=True)
def check_layout():
    for name in ('depth-model/frame','hexecutor/models','hexecutor/recovered','hexecutor/tools','hexecutor/depth-model'):
        if (ROOT/name).is_symlink():raise RuntimeError('retired resource alias: '+name)
    count=0
    for p in source_files():
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p));count+=1
    for p in (ROOT/'hexecutor/offline').glob('*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith('depth_model'):
                raise RuntimeError('executor imports a model: '+str(p))
    worker=(ROOT/'runtime/src/model_depth_pipeline.cpp').read_text()
    if 'FRAME_DEPTH_PYTHON' in worker or 'model_depth_bridge.py' in worker:raise RuntimeError('live path still launches Python')
    print(f'Layout, live native entry, and Python syntax checked ({count} files)',flush=True)
def check_model():
    run(sys.executable,'-m','unittest','discover','-s','hexecutor/tests','-p','test_resident_program.py')
    run(sys.executable,'-m','unittest','discover','-s','hexecutor/tests','-p','test_resident_image.py')
    script='''from pathlib import Path
import hashlib,json
from depth_model.adapters import NativeGraph
from check_kernels import load_inputs
from run_htp_emu import output_names
from hexecutor.paths import workspace
p=workspace()
for model in ('primary','alternate'):
 d=p.model_data/'recovered'/model
 with NativeGraph(p.native/f'build/resident-{model}.qdrx',d/'constants/constant_pool_5.bin') as g:
  out=g.run(load_inputs(p.native/'build/native-smoke'))
  expected={k.removesuffix('.raw'):v for k,v in json.loads((d/'emulator_smoke.json').read_text()).items()}
  for i,n in enumerate(output_names(d)):
   assert hashlib.sha256(out[i].tobytes()).hexdigest()==expected[n]['sha256'],(model,n)
 print(model,'native host model smoke: PASS')
'''
    run(sys.executable,'-c',script)
def main():
    command=sys.argv[1] if len(sys.argv)>1 else 'help'
    if command=='help':
        print('make build-host    Build native C++ host, worker, and portable tests\n'
              'make check         C++ tests, artifact policy, docs, layout, and geometry\n'
              'make check-model   Optional model-dependent native reader/reference checks\n'
              'make build-runtime Build the Linux/Vulkan headset app and native worker\n'
              'make build-dsp     Build basic Hexagon targets (toolchain required)\n'
              'make audit         Audit artifacts across Git history\n'
              'make run SCRIPT=path/to/tool.py ARGS="..."  Run offline Python tools\n'
              'FRAME_DEPTH_DATA, FRAME_DEPTH_ARTIFACTS, FRAME_DEPTH_PROGRAMS override local resources.')
    elif command=='check-docs':check_docs()
    elif command=='check-layout':check_layout()
    elif command=='check-model':check_model()
    elif command=='check':
        check_layout();check_docs();run(sys.executable,'tools/test_model_artifact_policy.py');run(sys.executable,'tools/model_artifact_policy.py','--history')
        run('c++','-O2','-std=c++17','-Iruntime/include','runtime/tools/test_model_depth_geometry.cpp','-o','build/test_geometry');run('build/test_geometry')
        run('c++','-O2','-std=c++17','-Iruntime/include','runtime/tools/test_model_temporal.cpp','runtime/src/model_temporal.cpp','-o','build/test_temporal');run('build/test_temporal')
    elif command=='run':
        if len(sys.argv)<3:raise ValueError('provide a Python script path')
        run(sys.executable,*sys.argv[2:])
    else:raise ValueError('unknown command: '+command)
if __name__=='__main__':main()
