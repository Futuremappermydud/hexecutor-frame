#!/usr/bin/env python3
"""Persistent Model depth runner for the native point-cloud worker.

The private inherited socket carries two already rectified/resized luma images
plus the rigid motion since the previous pair, and returns plane-0 float disparity. No raw DMA, files per frame or shell calls.

Backends:
  hybrid    validated v5 package: emulator graph with HVX/HMX convolution.
  prepared  NPRG v2 prepared executor: every command of the model in one
            resident DSP call (exact reference engine, ~15.5 s per pair).
"""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import socket
import struct
import sys
import time

# Pinned artifacts for the prepared backend, all validated on Frame 2026-11-01:
# full-model smoke/random outputs equal the emulator sha256 for both models.
PREPARED_ENGINE='frame/build/qd_graph_dsp-scalar-dynamic-tcm-fallback-l2-blocked-glue-fast-spill-native-1x1-native-3x3-direct-range-prepared-reference-vtcm-lease.elf'
PREPARED_BOOTSTRAP='frame/build/qd_graph_boot-c0-preserve-vtcm-guarded-entry.elf'
PREPARED_SHA256={
    PREPARED_ENGINE:'c432a7140ac633d9e20366ac8d801de1b42acdd2bf21dc2fdde65d415e9ce13b',
    PREPARED_BOOTSTRAP:'628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147',
    'frame/build/qd_loader.elf':'828ca0638750a6d8b69b3aaa536cb6c23508ce821b2333e270b455c321414572',
    'frame/native_program/examples/primary-v2.nprg':'c6ae3022dbbdab04f474751bd987e9bfc0547337d5e9b95becb2e100eed86013',
    'frame/native_program/examples/alternate-v2.nprg':'21953104d4960e19dff0f61116dd787e2eb533ef1e4c8eeeedfba197787f69a4',
    'frame/build/resident-primary.qdrx':'dc8fcddcf4876d11f7a717ec4f111e1d3d3a40d4f3163758b590234e0b50b92b',
    'frame/build/resident-alternate.qdrx':'54b2949bbb25655ee459c903432f67cacd26e66519854dcb9745a91777eb7298',
}
# QNN run-list engine, validated on Frame 2026-11-02 (claim_20261102i): primary
# smoke+random exact vs the hw oracle, 22.7 ms steady. Run list is primary-only.
QNN_ENGINE='frame/build/ab/claim_20261102i/eng-claim.elf'
QNN_BOOTSTRAP='frame/build/ab/claim_20261102i/boot-claim.elf'
PREPARED_SHA256[QNN_ENGINE]='678611d0d66eeeb28b01d76f403494cfe206babd2cd3b7fe4192417608629068'
PREPARED_SHA256[QNN_BOOTSTRAP]='cc30661b9496c489fcf6aad3fd79ad6009ab320f76e69edc3bc0aa319e0f798e'


def receive(sock, size, allow_eof=False):
    data=bytearray(size);view=memoryview(data);offset=0
    while offset<size:
        n=sock.recv_into(view[offset:])
        if not n:
            if allow_eof and offset==0:return None
            raise EOFError('incomplete Model depth request')
        offset+=n
    return data


@contextmanager
def hybrid_session(root,model):
    """Yields an infer(images, hidden, transforms) callable on the v5 package."""
    import sim_trace
    from fast_depth import NativeEmulator
    from frame_hvx import HvxConvolution
    from frame_transport import FrameTransport
    from run_frame import verify_package
    from run_htp_emu import execute
    # Verifies the installed package's complete manifest before touching the DSP.
    verify_package()
    recovered=root/'recovered'/model
    graph=sim_trace.load_graph(recovered);nodes=sim_trace.execution_nodes(graph)
    emu=NativeEmulator(graph,(recovered/'constants/constant_pool_5.bin').read_bytes(),{})
    transport=FrameTransport(root/'frame/build/qd_loader.elf',root/'frame/build/qd_conv.elf',10,use_hmx=True)
    with transport:
        emu.conv_backend=HvxConvolution(transport,direct=True)
        def infer(images,hidden,transforms):
            outputs=execute(emu,nodes,{0:images[0],1:images[1],2:hidden,3:transforms})
            return outputs[1],outputs[0]
        yield infer


@contextmanager
def prepared_session(root,model,engine=PREPARED_ENGINE,bootstrap=PREPARED_BOOTSTRAP):
    """Yields an infer(...) callable on a prepared executor engine."""
    from resident_dsp import ResidentDSP
    program=f'frame/native_program/examples/{model}-v2.nprg'
    image=f'frame/build/resident-{model}.qdrx'
    # Every pinned artifact is checked before the DSP lock is taken.
    for name in (engine,bootstrap,'frame/build/qd_loader.elf',program,image):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=PREPARED_SHA256[name]:
            raise ValueError(f'prepared artifact {name} does not match its validated sha256')
    with ResidentDSP(root/image,root/'recovered'/model/'constants/constant_pool_5.bin',
                     engine=root/engine,bootstrap=root/bootstrap,
                     prepared_program=root/program) as dsp:
        if not dsp.preserve_vtcm:raise ValueError('preserving bootstrap required')
        def infer(images,hidden,transforms):
            outputs=dsp.run([images[0],images[1],hidden,transforms])
            trace=dsp.last.get('native_direct_trace',[])
            if not dsp.last.get('native_prepared_program') or len(trace)!=16 or trace[8]!=dsp.commands:
                raise RuntimeError(f'prepared run did not execute the full model: {trace}')
            return outputs[1],outputs[0]
        yield infer
    print(f'[ModelDepth] Prepared session closed; VTCM lease {dsp.vtcm_lease}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fd',type=int,required=True)
    ap.add_argument('--package',type=Path,required=True)
    ap.add_argument('--model',choices=('primary','alternate'),default='primary')
    ap.add_argument('--backend',choices=('hybrid','prepared','prepared-qnn'),default='hybrid')
    ap.add_argument('--temporal',action='store_true',
                    help='Model-style hidden state: warp the previous disparity by the sent motion')
    args=ap.parse_args();root=args.package.resolve()
    sys.path[:0]=[str(root/'frame'),str(root/'tools'),str(Path(__file__).resolve().parent)]
    import os
    import numpy as np
    from model_temporal import ModelTemporalState
    os.sched_setaffinity(0,{7})
    # The Model's no-history inputs: zero state and an all-zero t (gate 0.8).
    # Identity transforms are not a state the Model runtime produces.
    zero_hidden=np.zeros((1,2,320,256),np.float32)
    zero_t=np.zeros((1,2,4,4),np.float32)
    temporal=ModelTemporalState() if args.temporal else None
    # The engine exits by itself after ~36.9 s without a command, and dsp_service
    # then stays at control (2,0,8) until reboot. Never let it get there: close
    # the whole session after an idle gap and reopen on the next request.
    idle_close=float(os.environ.get('FRAME_DEPTH_IDLE_CLOSE_S','15'))
    import contextlib,select
    stack=contextlib.ExitStack()
    infer=None
    def open_session():
        if args.backend=='prepared-qnn':
            if args.model!='primary':ap.error('prepared-qnn has a captured run list for primary only')
            return stack.enter_context(prepared_session(root,args.model,QNN_ENGINE,QNN_BOOTSTRAP))
        return stack.enter_context((prepared_session if args.backend=='prepared' else hybrid_session)(root,args.model))
    with socket.socket(fileno=args.fd) as sock,stack:
        infer=open_session()
        print(f'[ModelDepth] {args.backend} backend ready ({args.model}, {root}, temporal={bool(temporal)}, idle close {idle_close:g} s)',flush=True)
        sock.sendall(struct.pack('<4I',0x36524451,256,320,3))
        last_request=time.monotonic()
        while True:
            if infer is not None and not select.select([sock],[],[],1.0)[0]:
                if time.monotonic()-last_request>=idle_close:
                    stack.close();infer=None
                    if temporal is not None:temporal.reset()
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
            begin=time.monotonic()
            if infer is None:
                infer=open_session()
                print(f'[ModelDepth] DSP session reopened in {(time.monotonic()-begin)*1000:.0f} ms',flush=True)
            if temporal is not None:
                flags=motion_block[0]
                if flags&2:temporal.reset()
                motion=((np.array(motion_block[1:10]).reshape(3,3),np.array(motion_block[10:13]))
                        if flags&1 else None)
                hidden,transforms=temporal.inputs(motion)
            else:hidden,transforms=zero_hidden,zero_t
            output,uncertainty=infer(images,hidden,transforms)
            output=output.reshape(2,height,width)
            if temporal is not None:
                temporal.save(output)
                if sequence%25==1:print(f'[ModelDepth] temporal {temporal.stats}',flush=True)
            ms=(time.monotonic()-begin)*1000
            # Plane 0 = left disparity (outer.pte names the outputs disparity/confidence);
            # plane 1 (right) feeds only the temporal state.
            disparity=np.ascontiguousarray(output[0],dtype='<f4')
            sock.sendall(struct.pack('<3If',0x364f4451,sequence,width*height,ms))
            sock.sendall(disparity.tobytes())
            # v3: plane-0 uint8 uncertainty (the outputs' 'confidence' head; high at depth edges).
            sock.sendall(np.ascontiguousarray(uncertainty.reshape(2,height,width)[0],dtype=np.uint8).tobytes())
            last_request=time.monotonic()


if __name__=='__main__':
    try:main()
    except (BrokenPipeError,ConnectionResetError):pass
    except KeyboardInterrupt:sys.exit(130)
