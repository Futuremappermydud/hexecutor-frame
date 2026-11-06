#!/usr/bin/env python3
"""Per-node-type wall time of warm HMX inference (wraps NativeEmulator.run_node)."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'frame'),str(ROOT/'tools')]
import fast_depth
import run_frame

stats=defaultdict(lambda:[0,0.0])
original=fast_depth.NativeEmulator.run_node
def timed(self,node):
    start=time.perf_counter();original(self,node)
    s=stats[node['type_name']];s[0]+=1;s[1]+=time.perf_counter()-start
fast_depth.NativeEmulator.run_node=timed

if __name__=='__main__':
    output=Path(sys.argv[1]);repeat=sys.argv[2] if len(sys.argv)>2 else '3'
    sys.argv=['run_frame.py','--smoke','--compute-engine','hmx','--repeat',repeat,'--output',str(output)]+sys.argv[3:]
    try:run_frame.main()
    except SystemExit:pass
    n=int(repeat)
    rows=sorted(((v[1]/n*1000,v[0]//n,k) for k,v in stats.items()),reverse=True)
    for ms,c,k in rows:print(f'{ms:9.3f} ms {c:6d} {k}')
    print(f'total {sum(r[0] for r in rows):.3f} ms per inference (includes cold run)')
    (output/'node-profile.json').write_text(json.dumps([dict(ms=r[0],count=r[1],type=r[2]) for r in rows],indent=1))
