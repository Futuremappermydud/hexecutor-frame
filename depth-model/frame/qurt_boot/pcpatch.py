#!/usr/bin/env python3
"""pcpatch.py in.mbn out.mbn ADDR   -- ADDR is VA or PA; writes `jump self` (0x5800c000) there"""
import struct,sys
d=bytearray(open(sys.argv[1],'rb').read()); a=int(sys.argv[3],0)
phoff,=struct.unpack_from('<I',d,28); n,=struct.unpack_from('<H',d,44)
for i in range(n):
    t,off,va,pa,fs,ms,fl,al=struct.unpack_from('<8I',d,phoff+32*i)
    if t!=1 or not fs: continue
    for base in (va,pa):
        if base<=a<base+fs:
            o=off+a-base; print('seg',i,'file off',hex(o),'old',hex(struct.unpack_from('<I',d,o)[0]))
            struct.pack_into('<I',d,o,0x5800c000); open(sys.argv[2],'wb').write(d); sys.exit(0)
sys.exit('addr not found')
