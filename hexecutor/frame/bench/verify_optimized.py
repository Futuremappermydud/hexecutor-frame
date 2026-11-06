#!/usr/bin/env python3
"""Validate the optimized full graph on hardware against all simulator probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'frame')]
from fast_depth import NativeEmulator
from frame_hvx import HvxConvolution
from frame_transport import FrameTransport
from run_htp_emu import execute,output_names
from check_kernels import load_inputs
import sim_trace
from stereo_probes import make


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--report',type=Path,required=True);ap.add_argument('--hmx',action='store_true');ap.add_argument('--cpu',type=int,default=7);args=ap.parse_args()
    if args.cpu>=0:os.sched_setaffinity(0,{args.cpu})
    directory=ROOT/'frame/build/native-probes';make(directory.resolve())
    refs=json.loads((ROOT/'reference/simulator_output_hashes.json').read_text())['models'];report={}
    for model in ('primary','alternate'):
        recovered=ROOT/'recovered'/model;graph=sim_trace.load_graph(recovered);nodes=sim_trace.execution_nodes(graph)
        emu=NativeEmulator(graph,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
        report[model]={}
        with FrameTransport(ROOT/'frame/build/qd_loader.elf',ROOT/'frame/build/qd_conv.elf',10,use_hmx=args.hmx) as transport:
            backend=HvxConvolution(transport,direct=True);emu.conv_backend=backend
            for case,expected in refs[model].items():
                begin=time.monotonic();out=execute(emu,nodes,load_inputs(directory/case))
                hashes={name+'.raw':hashlib.sha256(out[i].tobytes()).hexdigest() for i,name in enumerate(output_names(recovered))}
                assert hashes==expected,(model,case,hashes,expected)
                report[model][case]={'equal':True,'seconds':time.monotonic()-begin,'hashes':hashes}
                print(model,case,'PASS',round(report[model][case]['seconds'],4),flush=True)
            inputs=load_inputs(ROOT/'frame/build/depth-inputs/random');out=execute(emu,nodes,inputs)
            expected=json.loads((recovered/'emulator_random.json').read_text())
            for i,name in enumerate(output_names(recovered)):
                assert hashlib.sha256(out[i].tobytes()).hexdigest()==expected[name]['sha256'],(model,'random',name)
            report[model]['random']={'equal':True}
            report[model]['quantized_convolutions']=backend.quantized_calls
            report[model]['hmx_convolutions']=transport.hmx_calls
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
        del emu


if __name__=='__main__':main()
