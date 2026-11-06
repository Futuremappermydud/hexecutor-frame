#!/usr/bin/env python3
"""Run an image stereo pair through the Frame's hybrid ARM/HVX depth backend."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from model_depth import IMAGE_SHAPE, STATE_SHAPE, TRANSFORM_SHAPE, checked, load_image, save_result
from run_htp_emu import output_names


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--left',type=Path,required=True)
    ap.add_argument('--right',type=Path,required=True)
    ap.add_argument('--model',choices=['primary','alternate'],default='primary')
    ap.add_argument('--hidden',type=Path)
    ap.add_argument('--transforms',type=Path)
    ap.add_argument('--resize',action='store_true')
    ap.add_argument('--repeat',type=int,default=1)
    ap.add_argument('--compute-engine',choices=('hvx','hmx'),default='hmx')
    ap.add_argument('--cpu',type=int,default=7)
    ap.add_argument('--no-clock',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    try:
        left,right=(load_image(p,args.resize) for p in (args.left,args.right))
        hidden=(np.zeros(STATE_SHAPE,np.float32) if args.hidden is None else
                checked(np.load(args.hidden,allow_pickle=False),STATE_SHAPE,np.float32,'hidden'))
        transforms=(np.tile(np.eye(4,dtype=np.float32),(1,2,1,1)) if args.transforms is None else
                    checked(np.load(args.transforms,allow_pickle=False),TRANSFORM_SHAPE,np.float32,'transforms'))
    except (ValueError,OSError,ImportError) as exc:
        ap.exit(2,f'depth-model: {exc}\n')
    args.output.mkdir(parents=True,exist_ok=True)
    inputs=args.output/'inputs';inputs.mkdir(exist_ok=True)
    for name,value in zip(('x','y','hidden','t'),(left,right,hidden,transforms)):
        value.tofile(inputs/f'{name}.raw')
    command=[sys.executable,str(ROOT/'frame/run_frame.py'),'--model',args.model,
             '--inputs',str(inputs.resolve()),'--output',str(args.output.resolve()),'--repeat',str(args.repeat),'--compute-engine',args.compute_engine,'--cpu',str(args.cpu)]
    if args.no_clock:command.append('--no-clock')
    subprocess.run(command,check=True)
    hardware=json.loads((args.output/'report.json').read_text())
    names=output_names(ROOT/'recovered'/args.model)
    aux=np.fromfile(args.output/f'{names[0]}.raw',np.uint8).reshape(STATE_SHAPE)
    floating=np.fromfile(args.output/f'{names[1]}.raw',np.float32).reshape(STATE_SHAPE)
    result={'auxiliary_u8':aux,'float_output':floating,
            'auxiliary_dequantized':(aux.astype(np.float32)-136)*np.float32(0.05533868819475174),
            'seconds':hardware['runs'][-1]['seconds']}
    (args.output/'hardware-report.json').write_text(json.dumps(hardware,indent=2)+'\n')
    save_result(result,args.output,{'model':args.model,'backend':'Frame ARM + Hexagon '+args.compute_engine.upper(),
                                  'context_sha256':hardware['context_sha256'],
                                  'left':str(args.left.resolve()),'right':str(args.right.resolve()),
                                  'zero_hidden_placeholder':args.hidden is None,
                                  'identity_transforms_placeholder':args.transforms is None,
                                  'hardware_report':'hardware-report.json'})
    print(f'Saved {args.output / "outputs.npz"}')


if __name__=='__main__':main()
