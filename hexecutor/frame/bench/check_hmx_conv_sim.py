"""Check simulated HMX+HVX output against independently accumulated int64 cases."""
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]/'build'
expected=np.load(root/'hmx-conv-expected.npz')
raw=(root/'hmx-conv-output.bin').read_bytes();offset=0
for case in range(sum(key.endswith('_pre') for key in expected.files)):
    pre=expected[f'{case}_pre'];q=expected[f'{case}_q'];wp=int(expected[f'{case}_owp'])
    h,w,n=pre.shape;size=h*wp*n
    got=np.frombuffer(raw,np.uint8,size,offset).reshape(h*wp//32,n//32,32,32)
    got=got.transpose(0,2,1,3).reshape(h,wp,n)[:,:w]
    want=np.clip((pre.astype(np.int64)*q[:,1]+8192)>>14,0,255).astype(np.uint8)
    np.testing.assert_array_equal(got,want)
    print('HMX + fused HVX quantization',case,tuple(expected[f'{case}_x']),tuple(expected[f'{case}_w']),'PASS')
    offset+=size
assert offset==len(raw),(offset,len(raw))
