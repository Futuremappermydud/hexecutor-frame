#!/usr/bin/env python3
"""Export both recovered graphs and the freestanding callback as a Frame package."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frame'))
from frame_transport import load_callback,FIRMWARE_SHA256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'frame/dist/depth-model-frame.zip')
    args=ap.parse_args()
    callback=ROOT/'frame/build/qd_hvx.elf'
    _,callback_sha=load_callback(callback)
    files={}
    tools=('htp_emu.py','sim_trace.py','check_kernels.py','run_htp_emu.py','prepare_inputs.py','stereo_probes.py')
    for name in tools:files['tools/'+name]=(ROOT/'tools'/name).read_bytes()
    for name in ('run_frame.py','frame_hvx.py','frame_transport.py','hvx_protocol.py','local_transport.py',
                 'probe_hvx.py','test_local.py','qd_hvx.c','qd_protocol.h','qd_link.ld','Makefile','Dockerfile','README.md'):
        files['frame/'+name]=(ROOT/'frame'/name).read_bytes()
    files['frame/build/qd_hvx.elf']=callback.read_bytes()
    for name in ('local-tests.json','local-full-models.json','hvx-probe.json',
                 'hardware-primary-smoke.json','hardware-primary-random.json',
                 'hardware-alternate-smoke.json','hardware-alternate-random.json',
                 'hardware-interrupt.json','post-run-state.json','hardware-validation.json'):
        path=ROOT/'frame/build'/name
        if path.exists():files['frame/validation/'+name]=path.read_bytes()
    files['requirements.txt']=b'numpy>=1.24\n'
    files['run.py']=b'#!/usr/bin/env python3\nfrom frame.run_frame import main\nmain()\n'
    files['reference/simulator_output_hashes.json']=(ROOT/'reference/simulator_output_hashes.json').read_bytes()
    models={}
    for model in ('primary','alternate'):
        d=ROOT/'recovered'/model
        for name in ('graph.json.gz','execution_order.json','kernel_calls.json','emulator_smoke.json','emulator_random.json'):
            files[f'recovered/{model}/{name}']=(d/name).read_bytes()
        files[f'recovered/{model}/constants/constant_pool_5.bin']=(d/'constants/constant_pool_5.bin').read_bytes()
        files[f'metadata/{model}.qnn.json']=(ROOT/f'metadata/{model}.qnn.json').read_bytes()
        models[model]={'context_sha256':json.loads((d/'execution_order.json').read_text())['context_sha256']}
    manifest={'format':'depth-model-frame-v1','models':models,
              'execution':'ARM graph operations plus Hexagon HVX128 convolution dot products',
              'frame_firmware_sha256':FIRMWARE_SHA256,'callback_sha256':callback_sha,
              'hardware_validation':'See frame/README.md and validation reports; package format is not a hardware-equivalence claim.',
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in sorted(files.items())}}
    files['frame-package.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo('depth-model-frame/'+name,date_time=(2026,10,25,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(0o755 if name=='run.py' else 0o644)<<16
            z.writestr(info,data)
    print(f'{args.output}: {args.output.stat().st_size} bytes, sha256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}')


if __name__=='__main__':main()
