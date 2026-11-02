import sys, numpy as np
raw=open(sys.argv[1],'rb').read(); R=256+2048+1024+2048; n=len(raw)//R
fi=np.zeros((n,64),np.int64); A=np.zeros((n,64,32),np.int64); W=np.zeros((n,32,32),np.int64); C=np.zeros((n,64,32),np.int64)
for t in range(n):
    r=raw[t*R:(t+1)*R]
    fi[t]=np.frombuffer(r[:256],'<i4')
    A[t]=np.frombuffer(r[256:2304],'u1').reshape(64,32)
    b=np.frombuffer(r[2304:3328],'i1').reshape(8,32,4)   # [k/4][n][k%4]
    W[t]=b.transpose(0,2,1).reshape(32,32)              # [k][n]
    C[t]=np.frombuffer(r[3328:],'u1').reshape(64,32)
acc=np.einsum('tpk,tkn->tpn',A,W)+fi[:,None,32:]
sc=fi[:,None,:32]; e=(sc>>10)&31; m=1024+(sc&1023); pre=20-e
assert ((sc>>15)==0x80).all()
x=acc
R1=np.clip((((x>>pre)*m)+8192)>>14,0,255)
R2=np.clip(((x*m)+(1<<(pre+13)))>>(pre+14),0,255)
for name,f in [('two-stage floor',R1),('single round',R2)]:
    d=C-f; print(name,'exact frac',(d==0).mean(),'delta range',d.min(),d.max(), 'hist',{int(k):int(v) for k,v in zip(*np.unique(d,return_counts=True))})
d=C-R1; bad=np.argwhere(d!=0)
print('examples (acc,pre,M,got,R1,R2):')
for t,p,c in bad[:12]:
    print(int(x[t,p,c]),int(pre[t,0,c]),int(m[t,0,c]),int(C[t,p,c]),int(R1[t,p,c]),int(R2[t,p,c]))
np.savez('/tmp/hmxq.npz' if len(sys.argv)<3 else sys.argv[2],x=x,pre=np.broadcast_to(pre,x.shape),m=np.broadcast_to(m,x.shape),C=C)
