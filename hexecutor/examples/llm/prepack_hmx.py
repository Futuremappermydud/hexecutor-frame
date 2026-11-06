#!/usr/bin/env python3
"""One-time GGUF Q4_0 -> FP16 HMX tile conversion; never used during chat."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from npu_qwen import MODEL_DEFAULT, GGUF_PY_DEFAULT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=MODEL_DEFAULT)
    parser.add_argument('--gguf-python',type=Path,default=GGUF_PY_DEFAULT)
    parser.add_argument('--output',type=Path,default=MODEL_DEFAULT.parent/'Qwen3-0.6B-HMX-F16')
    parser.add_argument('--only',help='convert one matrix for a quick check')
    args=parser.parse_args()
    sys.path.insert(0,str(args.gguf_python))
    import gguf
    reader=gguf.GGUFReader(args.model,mode='r')
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={}
    started=time.monotonic()
    for tensor in reader.tensors:
        if len(tensor.shape)!=2 or tensor.name=='token_embd.weight' or (args.only and tensor.name!=args.only):
            continue
        if tensor.tensor_type!=gguf.GGMLQuantizationType.Q4_0:
            raise ValueError(f'{tensor.name} is not Q4_0')
        k,n=map(int,tensor.shape)
        if k%32 or n%32:
            raise ValueError(f'{tensor.name} has unsupported shape {tensor.shape}')
        target=args.output/(tensor.name+'.hmx-f16')
        expected=k*n*2
        if target.exists() and target.stat().st_size==expected:
            metadata[tensor.name]={'k':k,'n':n,'bytes':expected}
            continue
        temporary=target.with_suffix(target.suffix+'.part')
        with temporary.open('wb') as output:
            for row in range(0,n,1024):
                count=min(1024,n-row)
                values=gguf.dequantize(tensor.data[row:row+count],tensor.tensor_type).astype('<f2')
                if values.shape!=(count,k):
                    raise ValueError(f'{tensor.name}: unexpected dequantized shape {values.shape}')
                packed=(values.reshape(count//32,32,k//32,16,2)
                         .transpose(0,2,3,1,4).copy(order='C'))
                packed.tofile(output)
        if temporary.stat().st_size!=expected:
            raise RuntimeError(f'{tensor.name}: wrong packed size')
        temporary.replace(target)
        metadata[tensor.name]={'k':k,'n':n,'bytes':expected}
        print(f'{tensor.name}: {expected/1048576:.1f} MiB; {time.monotonic()-started:.1f}s',flush=True)
    manifest={'source':str(args.model),'source_bytes':args.model.stat().st_size,
              'tensors':metadata,'seconds':time.monotonic()-started}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'complete: {len(metadata)} matrices, {sum(v["bytes"] for v in metadata.values())/1048576:.1f} MiB',flush=True)


if __name__=='__main__':main()
