exec(open('dag.py').read())
import heapq
WIN=int(__import__('os').environ.get('WIN','48'))
h0=honored_set(main,work)
# intra-worker conflicts: any conservative edge between two worker commands
wset={e[0] for e in work}
conf={}
for i in range(N):
    for p in preds[i]:
        if p in wset and i in wset:
            conf.setdefault(i,set()).add(p);conf.setdefault(p,set()).add(i)
# need: earliest main index that waits on a CP produced by this worker entry (direct), propagated backward along worker conflicts in original order
mpos_wait={}
for k,e in enumerate(main):
    if e[1]: mpos_wait.setdefault(e[1],[]).append(k)
need=[min(mpos_wait.get(e[2],[10**9])) if e[2] else 10**9 for e in work]
widx={e[0]:k for k,e in enumerate(work)}
for k in range(len(work)-1,-1,-1):
    for q in conf.get(work[k][0],()):
        j=widx[q]
        if j<k: need[j]=min(need[j],need[k])
def build(scale=lambda L,o,v,ix:1.0):
    """Simulate both lanes; the worker lane picks, among the next WIN unscheduled
    entries that are CP-ready and pass no earlier unscheduled conflicting entry,
    the one main needs first."""
    cnt=list(init);rel=[0.0]*len(init)
    t=[0.0,0.0];mp=0;sched=[];done=[False]*len(work);first=0
    def ready_time(e):
        w=e[1]
        return 0.0 if not w else (rel[w] if cnt[w]==0 else None)
    while mp<len(main) or len(sched)<len(work):
        # advance main as far as possible
        moved=False
        while mp<len(main):
            e=main[mp];rt=ready_time(e)
            if rt is None: break
            st=max(t[0],rt);t[0]=st+dur(0,e[0],scale)
            if e[2]: cnt[e[2]]-=1;rel[e[2]]=max(rel[e[2]],t[0])
            mp+=1;moved=True
        if len(sched)<len(work):
            while done[first]: first+=1
            best=None;blocked=set()
            k=first;seen=0
            while k<len(work) and seen<WIN:
                if done[k]: k+=1;continue
                e=work[k];seen+=1
                if not (conf.get(e[0],set())&blocked):
                    rt=ready_time(e)
                    if rt is not None and (best is None or (max(t[1],rt),need[k])<(max(t[1],best[1]),need[best[0]])):
                        if best is None or max(t[1],rt)<=max(t[1],best[1]) and need[k]<need[best[0]] or best is None: best=(k,rt)
                blocked.add(e[0]);k+=1
            if best is not None:
                k,rt=best;e=work[k]
                # prefer the entry main needs first among those ready by the time the lane is free
                st=max(t[1],rt);t[1]=st+dur(1,e[0],scale)
                if e[2]: cnt[e[2]]-=1;rel[e[2]]=max(rel[e[2]],t[1])
                done[k]=True;sched.append(e);moved=True
        if not moved: raise SystemExit('stuck mp=%d ws=%d'%(mp,len(sched)))
    return sched,t
if __name__=='__main__':
    sched,t=build()
    print('reordered model: main end %.2f worker end %.2f'%(t[0],t[1]))
    tt,w,b=simulate()
    print('qnn order model: %.2f'%max(tt))
    h1=honored_set(main,sched)
    print('honored lost:',len(h0-h1))
