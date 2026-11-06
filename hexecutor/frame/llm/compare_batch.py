#!/usr/bin/env python3
"""Locate the first difference between batched and serial CDSP inference."""
import numpy as np
from tokenizers import Tokenizer
from npu_qwen import QwenNPU


def layer(engine, token_ids, batched):
    p='blk.0.'
    print('layer',token_ids,batched,flush=True)
    x=engine.embed(token_ids if batched else token_ids[0])
    snapshots={'embed':np.array(x,copy=True)}
    z=engine.norm(p+'attn_norm.weight',x,1024)
    snapshots['attn_norm']=z.copy()
    q=engine.matvec(p+'attn_q.weight',z)
    k=engine.matvec(p+'attn_k.weight',z)
    v=engine.matvec(p+'attn_v.weight',z)
    snapshots['q']=q.copy();snapshots['k']=k.copy();snapshots['v']=v.copy()
    q=engine.norm(p+'attn_q_norm.weight',q,128,16)
    k=engine.norm(p+'attn_k_norm.weight',k,128,8)
    snapshots['q_norm']=q.copy();snapshots['k_norm']=k.copy()
    q,k=engine.rope(q,k,len(engine.history[0]))
    snapshots['q_rope']=q.copy();snapshots['k_rope']=k.copy()
    mixed=engine.attention(0,q,k,v)
    snapshots['attention']=mixed.copy()
    projected=engine.matvec(p+'attn_output.weight',mixed)
    snapshots['attn_out']=projected.copy()
    x=engine.add(x,projected)
    snapshots['residual']=x.copy()
    z=engine.norm(p+'ffn_norm.weight',x,1024)
    snapshots['ffn_norm']=z.copy()
    gate=engine.matvec(p+'ffn_gate.weight',z)
    up=engine.matvec(p+'ffn_up.weight',z)
    snapshots['gate']=gate.copy();snapshots['up']=up.copy()
    activation=engine.swiglu(gate,up)
    snapshots['swiglu']=activation.copy()
    down=engine.matvec(p+'ffn_down.weight',activation)
    snapshots['down']=down.copy()
    snapshots['final']=engine.add(x,down).copy()
    return snapshots


def main():
    tokenizer=Tokenizer.from_file(str(__import__('pathlib').Path.home()/'frame-chat/models/tokenizer.json'))
    ids=tokenizer.encode('<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False).ids
    print('ids',ids,flush=True)
    engine=QwenNPU()
    with engine:
        try:
            batched=layer(engine,ids,True)
            engine.history=[[] for _ in range(28)]
            serial=[layer(engine,[token],False) for token in ids]
            for name,value in batched.items():
                reference=np.stack([one[name] for one in serial])
                delta=np.max(np.abs(value-reference))
                relative=np.max(np.abs(value-reference)/np.maximum(np.abs(reference),0.1))
                print(name,delta,relative,flush=True)
        except BaseException as exc:
            print('inference exception',repr(exc),flush=True)
            raise


if __name__=='__main__':main()
