import cProfile, pstats, sys, time
from pathlib import Path
pkg=Path(sys.argv[1]); sys.path[:0]=[str(pkg/'frame'),str(pkg/'tools')]
sys.argv=['run_frame.py','--smoke','--output',sys.argv[2],'--backend','native-test']
import run_frame, htp_emu
import collections
per=collections.Counter(); cnt=collections.Counter()
_run=htp_emu.Emulator.run_node
def run_node(self,node):
    t=time.perf_counter(); _run(self,node); per[node['type_name']]+=time.perf_counter()-t; cnt[node['type_name']]+=1
htp_emu.Emulator.run_node=run_node
t0=time.perf_counter()
try: run_frame.main()
except SystemExit: pass
print('total', time.perf_counter()-t0)
for k,v in per.most_common(25): print(f'{v:7.3f}s {cnt[k]:5d} {k}')
