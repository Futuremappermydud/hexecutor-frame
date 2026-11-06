exec(open('sim.py').read().split("if __name__")[0])
import numpy as np
DT={'uint8':1,'int8':1,'uint16':2,'int16':2,'int32':4,'uint32':4,'float32':4,'float16':2}
def foot(ti):
    t=tf[ti];st=t.get('storage')
    if st is None: return {('T',ti)}
    if 'blocks' in st: return {(p,o//2048) for p,o in st['blocks']}
    p,o=st['direct'];d=t['shape']['max_dimensions'];n=DT.get(t['dtype'],4)
    for x in d: n*=x
    return {(p,b) for b in range(o//2048,(o+max(n,1)-1)//2048+1)}
tf=prog['tensor_features']
N=len(cmds)
R=[set().union(*[foot(i) for i in c[4]]) if c[4] else set() for c in cmds]
W=[set().union(*[foot(i) for i in c[5]]) if c[5] else set() for c in cmds]
preds=[set() for _ in range(N)]
lastw={};readers={}
for i in range(N):
    for b in R[i]:
        if b in lastw: preds[i].add(lastw[b])
    for b in W[i]:
        if b in lastw: preds[i].add(lastw[b])
        for r in readers.get(b,()):
            if r!=i: preds[i].add(r)
    for b in R[i]: readers.setdefault(b,set()).add(i)
    for b in W[i]: lastw[b]=i; readers[b]=set()
lane={}
for L,lst in ((0,main),(1,work)):
    for e in lst: lane[e[0]]=L
pos={}
for L,lst in ((0,main),(1,work)):
    for k,e in enumerate(lst): pos[e[0]]=(L,k)
cpprod={}
for L,lst in ((0,main),(1,work)):
    for k,e in enumerate(lst):
        if e[2]: cpprod.setdefault(e[2],[]).append((L,k))
def happens_before():
    # vc[L][k] = (count of main entries done, count of worker entries done) guaranteed before entry k of lane L starts
    lists=[main,work];vc=[[None]*len(main),[None]*len(work)]
    after=[[None]*len(main),[None]*len(work)]  # clock after completing
    ptr=[0,0];changed=True
    while ptr[0]<len(main) or ptr[1]<len(work):
        prog_=False
        for L in (0,1):
            while ptr[L]<len(lists[L]):
                k=ptr[L];e=lists[L][k];base=list(after[L][k-1]) if k else [0,0]
                ok=True
                if e[1]:
                    for (pl,pk) in cpprod.get(e[1],[]):
                        if after[pl][pk] is None: ok=False;break
                        base=[max(base[j],after[pl][pk][j]) for j in (0,1)]
                if not ok: break
                vc[L][k]=base;a=list(base);a[L]=k+1;after[L][k]=a;ptr[L]+=1;prog_=True
        if not prog_: raise SystemExit('deadlock')
    return vc
def honored_set(mainl,workl):
    """Edges (p,i) of the conservative DAG guaranteed by this run-list pair."""
    global main,work,pos,cpprod
    sm,sw=main,work
    main,work=mainl,workl
    pos={};cpprod={}
    for L,lst in ((0,main),(1,work)):
        for k,e in enumerate(lst):
            pos[e[0]]=(L,k)
            if e[2]: cpprod.setdefault(e[2],[]).append((L,k))
    vc=happens_before();hs=set()
    for i in range(N):
        L,k=pos[i]
        for p in preds[i]:
            pl,pk=pos[p]
            if vc[L][k][pl]>=pk+1: hs.add((p,i))
    main,work=sm,sw
    return hs
