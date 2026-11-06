#!/usr/bin/env python3
"""Host-only preflight for Frame probe packet and arena/pool ranges."""

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
import hashlib
import json
import struct
import sys
from pathlib import Path
HERE=(PATHS.native / 'kernels/adapters')
sys.path.insert(0,str(HERE))
sys.path.insert(0,str((PATHS.native)))
from run_adapter_frame_probe import check_oracle
MAGIC=0x31514151
FIXTURE_MAGIC=0x31585151

def overlap(a,n,b,m):return a<b+m and b<a+n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',type=Path,default=(PATHS.native / 'kernels/adapters/fixture-qdrx-1195.bin'));ap.add_argument('--output',type=Path,default=(PATHS.native / 'kernels/adapters/frame-probe-preflight.json'));a=ap.parse_args()
 raw=a.fixture.read_bytes();h=struct.unpack_from('<9I',raw);magic,seq,ih,iw,ci,co,blocks,wb,fb=h
 if magic!=FIXTURE_MAGIC or seq!=1195 or (ih,iw,ci,co)!=(8,64,128,128):raise ValueError('unsupported fixture')
 if (blocks,wb,fb)!=(32,ci*co,co*8) or len(raw)!=36+blocks*2048+wb+fb:raise ValueError('fixture sizes do not match command geometry')
 regions=[('fixture',0x10000,len(raw)),('packed_weights_copy',0x26000,wb),('fi_copy',0x2a000,fb),('pool1',0x40000,0x20000),('output_copy',0x90000,blocks*2048),('input_echo',0xa0000,blocks*2048),('block_tables',0x100000,2048),('vtcm_backup',0x300000,0x80000)]
 for i,(n,a0,an) in enumerate(regions):
  if a0+an>0x800000:raise ValueError(f'{n} outside arena')
  for m,b0,bn in regions[i+1:]:
   if overlap(a0,an,b0,bn):raise ValueError(f'arena regions overlap: {n}/{m}')
 packet=struct.pack('<10I',MAGIC,1,7,ih,iw,ci,co,1,0,0x700000)
 assert struct.unpack('<10I',packet)==(MAGIC,1,7,8,64,128,128,1,0,0x700000)
 simout=(PATHS.native / 'kernels/adapters/output-qdrx-1195.bin')
 if simout.exists():
  simraw=simout.read_bytes();bound=check_oracle(raw,simraw[36:])
  if bound[0]<0 or bound[1]>1 or bound[2]!=552:raise ValueError(f'Frame oracle differs from simulator oracle: {bound}')
 else:raise FileNotFoundError('expected known simulator output for host oracle check')
 report={'passed':True,'offline_only':True,'oracle_rechecked_simulator_output':True,'simulator_native_minus_integer_bound':list(bound[:2]),'simulator_different_values':bound[2],'fixture_sha256':hashlib.sha256(raw).hexdigest(),'fixture_bytes':len(raw),'packet_bytes':len(packet),'packet_words':list(struct.unpack('<10I',packet)),'regions':regions,'all_arena_ranges_disjoint':True,'aligned_staged_weight_offset':0x26000,'aligned_staged_fi_offset':0x2a000}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print('PASS host-only Frame probe fixture/packet/range preflight')
if __name__=='__main__':main()
