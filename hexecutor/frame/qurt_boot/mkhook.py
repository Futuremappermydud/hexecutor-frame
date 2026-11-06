import struct,sys
def build(hook,out):
    d=bytearray(open('p_base.mbn','rb').read())
    e=open('/Users/owenelliott/Documents/projects/spatial-stuff/depth-model/.work/stubtmp/'+('stub_pa.elf' if hook>>28==9 else 'stub_va.elf'),'rb').read()
    phoff,=struct.unpack_from('<I',e,28); t,o,va,pa,fs,ms,fl,al=struct.unpack_from('<8I',e,phoff)
    cave=e[o+4:o+fs]
    ph,=struct.unpack_from('<I',d,28); n,=struct.unpack_from('<H',d,44)
    def fo(a):
        for i in range(n):
            t,off,va,pa,fs,ms,fl,al=struct.unpack_from('<8I',d,ph+32*i)
            if t==1 and fs and va<=a<va+fs: return off+a-va
            if t==1 and fs and pa<=a<pa+fs: return off+a-pa
    def J(frm,to):
        off=((to-frm)>>2)&0x3fffff
        return 0x5800c000|((off&0x1fff)<<1)|(((off>>13)&0x1ff)<<16)
    C=0xfe000e40
    d[fo(C):fo(C)+len(cave)]=cave
    tgt = 0x9ca00e40 if hook>>28==9 else C
    struct.pack_into('<I',d,fo(hook),J(hook,tgt))
    open(out,'wb').write(d)
if __name__=='__main__': build(int(sys.argv[1],0),sys.argv[2])
