import socket, struct, array, mmap, hashlib
s=socket.socket(socket.AF_UNIX); s.settimeout(2); s.connect(b"\0dsp_service"); s.sendall(struct.pack("<I",4))
p,anc,_,_=s.recvmsg(64,socket.CMSG_SPACE(16)); fds=[]
for l,t,r in anc: fds.extend(array.array("i",r))
m=mmap.mmap(fds[1],0x2000000,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ,offset=0x80000000)
g=lambda o: struct.unpack_from("<I",m,o)[0]
print("ctrl", [g(0xA91000+4*i) for i in range(3)], "callback", hex(g(0xB12000+24)))
for name,off,n in (("loader",0x606000,4096),("payload",0x610000,0x8000),("arena",0x1800000,0x800000)):
    print(name, "nonzero" if any(m[off:off+n]) else "zero")
