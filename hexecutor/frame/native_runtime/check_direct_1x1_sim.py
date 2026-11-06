#!/usr/bin/env python3
"""Independent integer and staged-output oracle for the zero-copy 1195 SDK run."""
import hashlib
import json
import struct
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIX=HERE.parent/'native_adapter'/'fixture-qdrx-1195.bin'
STAGED=HERE.parent/'native_adapter'/'output-qdrx-1195.bin'
DIRECT=HERE/'direct-1x1-output.bin'
fixture=FIX.read_bytes();staged=STAGED.read_bytes();direct=DIRECT.read_bytes()
h=struct.unpack_from('<9I',fixture);magic,seq,height,width,ci,co,pages,wb,fb=h
if (magic,seq,height,width,ci,co,pages)!=(0x31585151,1195,8,64,128,128,32):
    raise SystemExit('unexpected 1195 fixture header')
if direct[:36]!=fixture[:36] or staged[:36]!=fixture[:36] or len(direct)!=len(staged):
    raise SystemExit('output/fixture shape mismatch')
pos=36;x=fixture[pos:pos+pages*2048];pos+=pages*2048
w=fixture[pos:pos+wb];pos+=wb;fi=struct.unpack('<%di'%(fb//4),fixture[pos:pos+fb])
got=direct[36:];old=staged[36:]
if got!=old:raise SystemExit('direct output differs from staged leaf output')
def value(data,y,x0,c):
    cb=c//32;idx=((y//8)*(width//8)+x0//8)*(ci//32)+cb
    return data[idx*2048+(y%8)*256+(x0%8)*32+c%32]
def weight(i,o):
    off=((o//32)*(ci//4)+i//4)*128+(o%32)*4+i%4
    v=w[off];return v if v<128 else v-256
deltas=[]
for y0 in range(height):
  for x0 in range(width):
    for o in range(co):
      acc=fi[(o//32)*64+32+o%32]
      for i in range(ci):acc+=value(x,y0,x0,i)*weight(i,o)
      m=fi[(o//32)*64+o%32]&0xffff;pre=34-((m>>10)&31)-14
      expected=max(0,min(255,(((acc>>pre)*(1024+(m&1023))+8192)>>14)))
      deltas.append(value(got,y0,x0,o)-expected)
if min(deltas)<0 or max(deltas)>1:raise SystemExit(f'int64 oracle delta outside [0,1]: [{min(deltas)},{max(deltas)}]')
report={
  'passed':True,'offline_only':True,'fixture_sequence':seq,
  'input_pages':pages,'output_pages':pages,'workspace_tensor_staging_bytes':0,
  'output_bytes':len(got),'oracle_delta':[min(deltas),max(deltas)],
  'direct_equals_staged':True,
  'direct_output_sha256':hashlib.sha256(direct).hexdigest(),
  'staged_output_sha256':hashlib.sha256(staged).hexdigest(),
  'fixture_sha256':hashlib.sha256(fixture).hexdigest(),
  'source_sha256':{
    'qd_native_direct.c':hashlib.sha256((HERE/'qd_native_direct.c').read_bytes()).hexdigest(),
    'qd_native_direct.h':hashlib.sha256((HERE/'qd_native_direct.h').read_bytes()).hexdigest(),
    'direct_1x1_sim.c':hashlib.sha256((HERE/'direct_1x1_sim.c').read_bytes()).hexdigest(),
  },
  'current_source_sha256':{
    'qd_native_direct.c':hashlib.sha256((HERE/'qd_native_direct.c').read_bytes()).hexdigest(),
    'qd_native_direct.h':hashlib.sha256((HERE/'qd_native_direct.h').read_bytes()).hexdigest()
  },
  'hardware_retest_required_for_current_core':True,
  'hardware_proof_v1_sha256':{
    'qd_native_direct.c':'6622e8126839081fa781736019f8605702e684a30bb6f75ad2b155676b955ce8',
    'qd_native_direct.h':'933c72b9f6f1072b40b78258460139c51dddcec3abc2110ffe6c41613c96469b'
  },
  'sdk_build_flags':['-mv75','-mhmx','-mhvx','-mhvx-length=128b','-O2','-G0','-ffreestanding','-fno-builtin','-fno-stack-protector'],
  'simulator':'Qualcomm Hexagon SDK 6.6.0.0 hexagon-sim -mv75 --quiet; simulator VTCM layout override 512 KiB (384 KiB active + 128 KiB reserved tail)',
  'checks':'QNN leaf [0,+1] int64 oracle; exact staged output; sim also checks source pools, VTCM outside active Y pages, input/A5 guards and zero-Fi unchanged'
}
(HERE/'direct_1x1_sim_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
