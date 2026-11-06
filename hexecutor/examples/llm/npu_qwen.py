#!/usr/bin/env python3
"""Qwen3-0.6B decode with all Transformer arithmetic on the Frame CDSP.

The ARM process owns GGUF parsing, tensor transfer, KV storage and token
selection. Matrix multiply runs on HMX; vector and attention operators run on
the CDSP scalar core. There is no CPU or GPU inference fallback.
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
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

HERE = (PATHS.repo / 'hexecutor/examples/llm')
FRAME = (PATHS.native)
sys.path.insert(0, str(FRAME))
from hexecutor.offline.frame_transport import FrameTransport

MODEL_DEFAULT = Path.home() / "frame-chat/models/Qwen3-0.6B-Q4_0.gguf"
GGUF_PY_DEFAULT = Path.home() / "frame-chat/vendor/llama.cpp/gguf-py"


class QwenNPU:
    def __init__(self, model=MODEL_DEFAULT, gguf_python=GGUF_PY_DEFAULT):
        sys.path.insert(0, str(gguf_python))
        import gguf
        self.gguf = gguf
        self.reader = gguf.GGUFReader(model, mode="r")
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.transport = None
        self.history = [[] for _ in range(28)]
        self.calls = 0
        self.wake_calls = 0
        self.call_seconds = 0.0
        self.matmul_seconds = 0.0
        self.math_seconds = 0.0
        self.rope_table = np.fromfile((PATHS.repo / 'hexecutor/examples/llm/rope_table.f32'), dtype='<f4').reshape(257,64,2)
        self.q4_tiles = os.environ.get('FRAME_LLM_Q4_TILES') == '1'
        self.packed_dir = Path(model).parent / ('Qwen3-0.6B-HMX-Q4' if self.q4_tiles else 'Qwen3-0.6B-HMX-F16')
        self.packed_cache = {}
        self._verify_model()

    def _verify_model(self):
        expected = {"blk.0.attn_q.weight": (1024, 2048),
                    "blk.0.ffn_down.weight": (3072, 1024),
                    "output.weight": (1024, 151936)}
        for name, shape in expected.items():
            tensor = self.tensors[name]
            if tuple(tensor.shape) != shape or tensor.tensor_type != self.gguf.GGMLQuantizationType.Q4_0:
                raise ValueError(f"Unexpected model tensor {name}: {tensor.shape}, {tensor.tensor_type}")

    def __enter__(self):
        payload = Path(os.environ.get("FRAME_LLM_PAYLOAD", FRAME / "build/hmx_q4_matvec.elf"))
        self.transport = FrameTransport(FRAME / "build/qd_loader.elf", payload, 20)
        self.transport.__enter__()
        self.wake_calls = 0
        return self

    def __exit__(self, *exc):
        if self.transport is not None:
            self.transport.__exit__(*exc)
            self.transport = None

    def command(self, op, fields, input_data=b"", weights=b"", output_count=0, weight_format=0):
        if self.transport is None:
            raise RuntimeError("NPU session is not open")
        if not isinstance(input_data, bytes):
            input_data = np.asarray(input_data, np.float32).tobytes()
        if not isinstance(weights, (bytes, memoryview)):
            weights = np.asarray(weights, np.float32).tobytes()
        if len(input_data)>0xa0000 or len(weights)>0x740000:
            raise ValueError("operator packet exceeds the shared arena")
        t = self.transport
        if self.wake_calls >= 128:
            t.stop_persistent()
            self.wake_calls = 0

        def prepare(seq):
            # Persistent DSP workers poll sequence at byte 8. Keep its old
            # value intact until the entire next packet is ready.
            t.arena[32:256]=bytes(224)
            t.arena[2048:2176]=bytes(128)
            struct.pack_into("<10I", t.arena, 0, 0x34514d48, 2, seq-1, op,
                             *(tuple(fields)+(0,)*(4-len(fields))),weight_format,(~seq)&0xffffffff)
            t.arena[0x20000:0x20000+len(input_data)]=input_data
            t.arena[0xc0000:0xc0000+len(weights)]=weights
            t.fence()
            struct.pack_into('<I',t.arena,8,seq)
            t.fence()

        def collect():
            status=struct.unpack_from("<32I",t.arena,2048)
            workers=[struct.unpack_from('<4I',t.arena,128+16*i) for i in range(3)]
            if output_count:
                output=np.frombuffer(bytes(t.arena[0x20000:0x20000+output_count*4]),np.float32)
            else:
                output=np.empty(0,np.float32)
            return status,output,workers

        start=time.monotonic()
        status, output, workers=t.submit(prepare, collect)
        self.wake_calls += 1
        elapsed=time.monotonic()-start
        self.calls+=1
        self.call_seconds+=elapsed
        if op==0:self.matmul_seconds+=elapsed
        else:self.math_seconds+=elapsed
        if status[1] or (op==0 and (status[2] or status[3]!=32 or status[7] or status[10] or status[11] or status[12]!=68)):
            raise RuntimeError(f"NPU op {op} call {self.calls} seq {t.sequence} failed: status={status} workers={workers}")
        return output

    def matvec(self, name, vector):
        tensor=self.tensors[name]
        k,n=map(int,tensor.shape)
        x=np.asarray(vector,np.float32)
        single=x.ndim==1
        if single:x=x.reshape(1,-1)
        if x.ndim!=2 or x.shape[1]!=k or not 1<=x.shape[0]<=32:
            raise ValueError(f"{name}: expected 1..32 rows of width {k}, got {x.shape}")
        batch=x.shape[0]
        packed=self._packed_matrix(name,k,n)
        parts=[]
        # K=1024 leaves room for 3072 output rows in the 8 MiB VTCM.
        # This cuts the Q/K and FFN projections, and the vocabulary head,
        # into far fewer HMX dispatches during single-token decoding.
        block_rows=3072 if k<=1024 else 1024
        for start in range(0,n,block_rows):
            rows=min(block_rows,n-start)
            packet=(memoryview(packed[start//32:(start+rows)//32]).cast('B') if packed is not None
                    else memoryview(tensor.data[start:start+rows]).cast('B'))
            parts.append(self.command(0,(k,rows,0,batch),x,packet,batch*rows,
                                      weight_format=2 if self.q4_tiles else int(packed is not None)).reshape(batch,rows))
        result=np.concatenate(parts,axis=1) if len(parts)>1 else parts[0]
        return result[0] if single else result

    def _packed_matrix(self,name,k,n):
        if name in self.packed_cache:
            return self.packed_cache[name]
        path=self.packed_dir/(name+('.hmx-q4' if self.q4_tiles else '.hmx-f16'))
        if not path.exists():
            if self.q4_tiles:
                raise FileNotFoundError(f'{name}: missing HMX Q4 tiles at {path}')
            self.packed_cache[name]=None
            return None
        size=k*n*640//1024 if self.q4_tiles else k*n*2
        if path.stat().st_size!=size:
            raise ValueError(f'{name}: wrong HMX packed size')
        mapped=(np.memmap(path,dtype='u1',mode='r',shape=(n//32,k//32,640)) if self.q4_tiles
                else np.memmap(path,dtype='<u2',mode='r',shape=(n//32,k//32,16,32,2)))
        self.packed_cache[name]=mapped
        return mapped

    def embed(self, token_id):
        tensor=self.tensors["token_embd.weight"]
        ids=np.asarray(token_id,dtype=np.int32)
        single=ids.ndim==0
        ids=ids.reshape(-1)
        if len(ids)>32 or np.any(ids<0) or np.any(ids>=int(tensor.shape[1])):
            raise ValueError(f"bad token IDs {ids}")
        raw=b"".join(tensor.data[int(i)].tobytes() for i in ids)
        result=self.command(6,(1024,len(ids)),b"",raw,len(ids)*1024).reshape(len(ids),1024)
        return result[0] if single else result

    def norm(self, name, vector, width, groups=1):
        weight=self.tensors[name].data
        x=np.asarray(vector,np.float32)
        total=x.size
        return self.command(1,(width,total//width),x,weight,total).reshape(x.shape)

    def add(self, a, b):
        return self.command(2,(np.asarray(a).size,),a,b,np.asarray(a).size).reshape(np.asarray(a).shape)

    def swiglu(self, gate, up):
        return self.command(3,(np.asarray(gate).size,),gate,up,np.asarray(gate).size).reshape(np.asarray(gate).shape)

    def rope(self, q, k, position):
        single=np.asarray(q).ndim==1
        if single:q=np.asarray(q).reshape(1,-1);k=np.asarray(k).reshape(1,-1)
        pairs=self.rope_table[position:position+q.shape[0]]
        rotated=self.command(4,(q.shape[0],),np.concatenate((q,k),axis=1),pairs,q.shape[0]*3072).reshape(-1,3072)
        return (rotated[0,:2048],rotated[0,2048:]) if single else (rotated[:,:2048],rotated[:,2048:])

    def attention(self, layer, q, k, v):
        single=np.asarray(q).ndim==1
        if single:q=q.reshape(1,-1);k=k.reshape(1,-1);v=v.reshape(1,-1)
        previous=self.history[layer]
        if len(previous)+len(q)>257:
            raise ValueError("NPU context limit is 257 tokens")
        history=(np.stack([np.stack((old_k,old_v)) for old_k,old_v in previous])
                 if previous else b"")
        joined=np.concatenate((q.reshape(-1),k.reshape(-1),v.reshape(-1)))
        result=self.command(5,(len(previous),len(q)),joined,history,len(q)*2048).reshape(-1,2048)
        previous.extend((np.array(kr,copy=True),np.array(vr,copy=True)) for kr,vr in zip(k,v))
        return result[0] if single else result

    def run_token(self, token_id, *, logits=True, progress=None):
        """Process one token on CDSP."""
        return self.run_batch([token_id],logits=logits,progress=progress)

    def run_batch(self, token_ids, *, logits=True, progress=None):
        """Process up to 32 causal tokens in one NPU pass."""
        position=len(self.history[0])
        if not 1<=len(token_ids)<=32 or position+len(token_ids)>257:
            raise ValueError("NPU context limit is 257 tokens")
        x=self.embed(token_ids)
        for layer in range(28):
            prefix=f"blk.{layer}."
            normalized=self.norm(prefix+"attn_norm.weight",x,1024)
            q=self.matvec(prefix+"attn_q.weight",normalized)
            k=self.matvec(prefix+"attn_k.weight",normalized)
            v=self.matvec(prefix+"attn_v.weight",normalized)
            q=self.norm(prefix+"attn_q_norm.weight",q,128,16)
            k=self.norm(prefix+"attn_k_norm.weight",k,128,8)
            q,k=self.rope(q,k,position)
            mixed=self.attention(layer,q,k,v)
            projected=self.matvec(prefix+"attn_output.weight",mixed)
            x=self.add(x,projected)
            normalized=self.norm(prefix+"ffn_norm.weight",x,1024)
            gate=self.matvec(prefix+"ffn_gate.weight",normalized)
            up=self.matvec(prefix+"ffn_up.weight",normalized)
            activated=self.swiglu(gate,up)
            down=self.matvec(prefix+"ffn_down.weight",activated)
            x=self.add(x,down)
            if progress:
                progress(layer+1,28)
        if not logits:
            return None
        x=self.norm("output_norm.weight",x,1024)
        return self.matvec("output.weight",x[-1])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,default=MODEL_DEFAULT)
    p.add_argument("--gguf-python",type=Path,default=GGUF_PY_DEFAULT)
    p.add_argument("--token-id",type=int,default=1000)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    started=time.monotonic()
    engine=QwenNPU(args.model,args.gguf_python)
    with engine:
        logits=engine.run_token(args.token_id,progress=lambda done,total:print(f"layer {done}/{total}",flush=True))
    report={"token_id":args.token_id,"greedy_next_token":int(np.argmax(logits)),
            "logit_sample":logits[:16].tolist(),"logit_min":float(np.min(logits)),
            "logit_max":float(np.max(logits)),"calls":engine.calls,
            "call_seconds":engine.call_seconds,"matmul_seconds":engine.matmul_seconds,
            "math_seconds":engine.math_seconds,"total_seconds":time.monotonic()-started}
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
