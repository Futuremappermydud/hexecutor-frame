#!/usr/bin/env python3
"""Offline depth replay/diagnostic bridge. Live inference uses frame_depth_worker.

The private inherited socket carries two already rectified/resized luma images
plus the rigid motion since the previous pair, and returns plane-0 float disparity. No raw DMA, files per frame or shell calls.

Backends:
  prepared-qnn  QNN run-list engine (~23 ms per pair, primary only). USE THIS.
  prepared      NPRG v2 prepared executor: every command of the model in one
                resident DSP call (exact reference engine, ~15.5 s per pair).
"""

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

import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import socket
import struct
import sys
import time

from depth_model.package import (PREPARED_ENGINE, PREPARED_BOOTSTRAP, PREPARED_SHA256,
                                 QNN_ENGINE, QNN_BOOTSTRAP, prepared_session)



def receive(sock, size, allow_eof=False):
    data=bytearray(size);view=memoryview(data);offset=0
    while offset<size:
        n=sock.recv_into(view[offset:])
        if not n:
            if allow_eof and offset==0:return None
            raise EOFError('incomplete Model depth request')
        offset+=n
    return data




def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fd',type=int,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--backend',choices=('prepared','prepared-qnn'),default='prepared-qnn')
    ap.add_argument('--spherical-deg',type=float,default=0,
                    help='model inputs are Model-style spherical at this deg/px (0 = resized pinhole)')
    ap.add_argument('--host-temporal',action='store_true',
                    help='each request carries hidden and t, warped by the native worker')
    ap.add_argument('--second-disparity',action='store_true',
                    help='append right disparity plane to each reply (protocol v4)')
    ap.add_argument('--close-after-reply',action='store_true',
                    help='close the cDSP session after each capture reply')
    args=ap.parse_args()
    if args.backend=='prepared-qnn' and args.model!='primary':
        ap.error('prepared-qnn has a captured run list for primary only; use --backend prepared for alternate')
    root=args.package.resolve()
    sys.path[:0]=[str(root/'frame'),str(root/'tools'),str((PATHS.repo / 'runtime/scripts'))]
    import os
    import numpy as np
    os.sched_setaffinity(0,{7})
    # The Model's no-history inputs: zero state and an all-zero t (gate 0.8).
    # Identity transforms are not a state the Model runtime produces.
    zero_hidden=np.zeros((1,2,320,256),np.float32)
    zero_t=np.zeros((1,2,4,4),np.float32)
    # The engine exits by itself after ~36.9 s without a command, and dsp_service
    # then stays at control (2,0,8) until reboot. Never let it get there: close
    # the whole session after an idle gap and reopen on the next request.
    idle_close=float(os.environ.get('FRAME_DEPTH_IDLE_CLOSE_S','15'))
    import contextlib,select
    stack=contextlib.ExitStack()
    infer=None
    def open_session():
        if args.backend=='prepared-qnn':
            return stack.enter_context(prepared_session(root,args.model,QNN_ENGINE,QNN_BOOTSTRAP))
        return stack.enter_context(prepared_session(root,args.model))
    with socket.socket(fileno=args.fd) as sock,stack:
        infer=open_session()
        print(f'[ModelDepth] {args.backend} backend ready ({args.model}, {root}, host temporal={args.host_temporal}, idle close {idle_close:g} s)',flush=True)
        sock.sendall(struct.pack('<4I',0x36524451,256,320,4 if args.second_disparity else 3))
        last_request=time.monotonic()
        while True:
            if infer is not None and not select.select([sock],[],[],1.0)[0]:
                if time.monotonic()-last_request>=idle_close:
                    stack.close();infer=None
                    print(f'[ModelDepth] idle {idle_close:g} s: DSP session closed cleanly; reopens on the next pair',flush=True)
                continue
            header=receive(sock,16,allow_eof=True)
            if header is None:break
            magic,sequence,width,height=struct.unpack('<4I',header)
            if magic!=0x36504451 or (width,height)!=(256,320):raise ValueError('invalid Model depth request')
            # flags (bit0 aligned motion, bit1 reset), R row-major, t, 3 pad words
            motion_block=struct.unpack('<I12f3I',receive(sock,64))
            pair=receive(sock,2*width*height)
            images=np.frombuffer(pair,np.uint8).reshape(2,height,width)
            if args.host_temporal:
                hidden=np.frombuffer(receive(sock,2*height*width*4),'<f4').reshape(1,2,height,width)
                transforms=np.frombuffer(receive(sock,32*4),'<f4').reshape(1,2,4,4)
            else:hidden,transforms=zero_hidden,zero_t
            begin=time.monotonic()
            if infer is None:
                infer=open_session()
                print(f'[ModelDepth] DSP session reopened in {(time.monotonic()-begin)*1000:.0f} ms',flush=True)
            output,uncertainty=infer(images,hidden,transforms)
            if np.asarray(output).size != 2*height*width or np.asarray(uncertainty).size != 2*height*width:
                raise ValueError('Model output shape mismatch')
            output=output.reshape(2,height,width)
            ms=(time.monotonic()-begin)*1000
            # Plane 0 = left disparity (outer.pte names the outputs disparity/confidence);
            # plane 1 (right) feeds the worker's temporal state.
            disparity=np.ascontiguousarray(output[0],dtype='<f4')
            sock.sendall(struct.pack('<3If',0x364f4451,sequence,width*height,ms))
            sock.sendall(disparity.tobytes())
            # v3: plane-0 uint8 uncertainty (the outputs' 'confidence' head; high at depth edges).
            sock.sendall(np.ascontiguousarray(uncertainty.reshape(2,height,width)[0],dtype=np.uint8).tobytes())
            if args.second_disparity:
                sock.sendall(np.ascontiguousarray(output[1],dtype='<f4').tobytes())
            last_request=time.monotonic()
            if args.close_after_reply:
                stack.close();infer=None
                print('[ModelDepth] capture reply complete; DSP session closed',flush=True)


if __name__=='__main__':
    try:main()
    except (BrokenPipeError,ConnectionResetError):pass
    except KeyboardInterrupt:sys.exit(130)
