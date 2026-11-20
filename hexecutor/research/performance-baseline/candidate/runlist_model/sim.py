# SPDX-License-Identifier: LGPL-3.0-or-later
exec(open(__import__('os').path.join(__import__('os').path.dirname(__file__) if '__file__' in globals() else '.','rl.py')).read())
import json
r=json.load(open('/tmp/frame-i-merge/depth-model/frame/build/ab/claim_20261102i/claim_primary.json'))
r=[x for x in r['runs'] if x['case']=='smoke'][-1]
avg={}
for x in r['handler_profile']:
    k=(x['opcode'],x['variant']);mc=x['count']&0xffffffff;wc=x['count']>>32
    if mc: avg[(0,)+k]=x['run_ms']/mc
    if wc: avg[(1,)+k]=x['validate_ms']/wc
OVH=0.00009  # ms per command outside handler (measured ~0.45ms/5068)
def dur(L,ix,scale):
    c=cmds[ix];k=(L,c[1],c[2]);d=avg.get(k,0.0)
    return d*scale(L,c[1],c[2],ix)+OVH
def simulate(scale=lambda L,o,v,ix:1.0,order=None):
    lists=[main,work];pos=[0,0];t=[0.0,0.0];cp=list(init);rel=[0.0]*len(init)
    waited=[0.0,0.0];busy=[0.0,0.0]
    crit=[]  # per list: (ix, wait)
    import heapq
    pending=[]  # cp decrements scheduled: (time, cp)
    # event-driven: each list proceeds; a wait on cp needs cp==0; cp time = max release time of its producers
    # since all producers known statically, compute release time when count hits 0.
    cnt=list(init);
    done_any=True
    while pos[0]<len(main) or pos[1]<len(work):
        progressed=False
        for L in (0,1):
            while pos[L]<len(lists[L]):
                ix,w,d,_e,_k=lists[L][pos[L]]
                if w and cnt[w]>0: break
                start=max(t[L], rel[w] if w else 0.0)
                waited[L]+=start-t[L]
                dd=dur(L,ix,scale);busy[L]+=dd
                t[L]=start+dd
                if d:
                    cnt[d]-=1; rel[d]=max(rel[d],t[L])
                pos[L]+=1;progressed=True
        if not progressed: raise SystemExit('deadlock')
    return t,waited,busy
if __name__=='__main__':
    t,w,b=simulate()
    print('model: main end %.2f wait %.2f busy %.2f | worker end %.2f wait %.2f busy %.2f'%(t[0],w[0],b[0],t[1],w[1],b[1]))
    ep=r['executor_profile'];hp=r['handler_profile']
    print('meas : main wall %.2f wait %.2f busy %.2f | worker wait %.2f busy %.2f'%(ep['run_ms'],ep['dispatch_ms'],sum(x['run_ms'] for x in hp),ep['validate_ms'],sum(x['validate_ms'] for x in hp)))
