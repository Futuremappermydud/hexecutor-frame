import sys, json, collections
from pathlib import Path
pkg=Path(sys.argv[1]); sys.path[:0]=[str(pkg/'frame'),str(pkg/'tools')]
import frame_hvx
rec=collections.Counter()
_acc=frame_hvx.HvxConvolution.accumulate
def acc(self,x,w,zp):
    rec[(x.shape,w.shape,str(x.dtype))]+=1
    return _acc(self,x,w,zp)
frame_hvx.HvxConvolution.accumulate=acc
sys.argv=['run_frame.py','--smoke','--output',sys.argv[2],'--backend','packed-cpu']
import run_frame
try: run_frame.main()
except SystemExit: pass
for k,v in sorted(rec.items(),key=lambda kv:-kv[1]): print(v,k)
