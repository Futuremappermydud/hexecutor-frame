"""Check native convolution geometry and quantify conversion differences."""
from pathlib import Path
import struct,json,sys
import numpy as np
BUILD=Path(__file__).resolve().parents[1]/'build'
expected=np.load(BUILD/'hmx-conv-expected.npz');raw=(BUILD/('hmx-native-cm-conv-output.bin' if '--cm' in sys.argv else 'hmx-native-conv-output.bin')).read_bytes();at=0;rows=[];seen={}
while at<len(raw):
    test,x32,size=struct.unpack_from('<3I',raw,at);at+=12
    if at+size>len(raw):raise ValueError('incomplete output')
    oh,ow,n=expected[f'{test}_pre'].shape;owp=int(expected[f'{test}_owp']);q=expected[f'{test}_q'].astype(np.int64)
    got=np.frombuffer(raw,np.uint8,size,at).reshape(oh,owp//32,n//32,32,32).transpose(0,1,3,2,4).reshape(oh,owp,n)[:,:ow].copy();at+=size
    want=np.clip((expected[f'{test}_pre'].astype(np.int64)*q[:,1]+8192)>>14,0,255).astype(np.uint8)
    delta=got.astype(np.int16)-want;row={'case':test,'x32':bool(x32),'values':got.size,'different':int(np.count_nonzero(delta)),'min':int(delta.min()),'max':int(delta.max())};rows.append(row);print(json.dumps(row))
    if delta.min()<0 or delta.max()>1:raise AssertionError('native convolution disagrees beyond calibrated conversion rounding')
    if test in seen:np.testing.assert_array_equal(seen[test],got)
    seen[test]=got
if len(seen)!=len([k for k in expected.files if k.endswith('_pre')]):raise AssertionError('missing convolution cases')
(BUILD/('hmx-native-cm-conv-sim.json' if '--cm' in sys.argv else 'hmx-native-conv-sim.json')).write_text(json.dumps({'cases':rows,'layouts_match':True,'conversion_difference_bound':[0,1]},indent=2)+'\n')
print('PASS convolution arithmetic/layout; conversion differences measured')
