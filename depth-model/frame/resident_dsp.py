"""Experimental whole-graph DSP backend over private cdsp_mem DDR.

The module must already be loaded at 128 MB / IOVA 0xa0000000. The transport
lock is held before inspecting or writing it. A marked preserving bootstrap
borrows VTCM only during the exclusive session and verifies its live snapshot
is restored before release. This never loads/unloads a module automatically
or changes the installed v5 path.
"""
import ctypes
import hashlib
import mmap
import os
from pathlib import Path
import struct
import time
import numpy as np
from cdsp_mem_check import require_clean_map
from frame_transport import FrameTransport,load_callback
from native_graph import INPUTS,OUTPUT_SHAPES

BASE=0xa0000000
SIZE=128<<20
MAGIC=0x31534451
FRAME=Path(__file__).resolve().parent

class ResidentDSP:
    def __init__(self,image,constants,engine=None,direct_hmx=False,bootstrap=None,prepared_program=None):
        self.image=Path(image).read_bytes();self.constants=Path(constants).read_bytes()
        self.header=struct.unpack_from('<64I',self.image)
        if self.header[0]!=int.from_bytes(b'QDRX','little'):raise ValueError('QDRX required')
        if hashlib.sha256(self.constants).digest()!=struct.pack('<8I',*self.header[16:24]):raise ValueError('constants mismatch')
        self.engine=Path(engine) if engine else FRAME/'build/qd_graph_dsp.elf'
        self.bootstrap=Path(bootstrap) if bootstrap else FRAME/'build/qd_graph_boot.elf'
        features=self.bootstrap.read_bytes().find(b'QDBOOT01\x01\x00\x00\x00')
        self.preserve_vtcm=features>=0
        self.l2_cache=b'QDL2BOOT02' in self.bootstrap.read_bytes()
        if self.l2_cache != (b'QDL2ENGN03' in self.engine.read_bytes()):
            raise ValueError('L2 cache bootstrap and engine features differ')
        self.direct_hmx=bool(direct_hmx)
        self.prepared_program=Path(prepared_program).read_bytes() if prepared_program else None
        self.transport=None;self.mem=None;self.cursor=1<<20;self.guards=[]

    def _alloc(self,size):
        off=(self.cursor+255)&~127
        end=(off+size+127)&~127
        self.guards.extend(((off-128,128),(end,128)))
        self.cursor=end+128
        if self.cursor>32<<20:raise ValueError('resident fixed data exceeds its region')
        return off

    def __enter__(self):
        # Before the engine writes anything: stale CDSP translations have
        # resolved chunks of this window to foreign memory.
        require_clean_map()
        self.transport=FrameTransport(FRAME/'build/qd_loader.elf',self.bootstrap,120)
        self.transport.__enter__()
        try:
            if int(Path('/sys/module/cdsp_mem/parameters/size_mb').read_text())!=128:raise ValueError('requires a private 128 MB cdsp_mem allocation')
            if int(Path('/sys/module/cdsp_mem/parameters/iova').read_text())!=BASE:raise ValueError('wrong CDSP IOVA')
            fd=os.open('/dev/cdsp_mem',os.O_RDWR)
            try:self.mem=mmap.mmap(fd,SIZE,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE)
            finally:os.close(fd)
            code,self.engine_sha256=load_callback(self.engine,BASE,1<<20)
            self.mem[:len(code)]=code
            image=self._alloc(len(self.image));self.mem[image:image+len(self.image)]=self.image
            self.sizes=list(self.header[24:31]);self.pools=[0]*7
            self.pools[5]=self._alloc(self.sizes[5])
            const=self.pools[5];self.mem[const:const+len(self.constants)]=self.constants
            prepared=0
            if self.prepared_program is not None:
                prepared=self._alloc(len(self.prepared_program))
                self.mem[prepared:prepared+len(self.prepared_program)]=self.prepared_program
            if self.cursor>16<<20:raise ValueError('immutable data exceeds cached region')
            self.cursor=16<<20
            for p,n in enumerate(self.sizes):
                if p!=5 and n:self.pools[p]=self._alloc(n)
            self.inputs=[self._alloc(n*np.dtype(dt).itemsize) for _,dt,n in INPUTS]
            self.input_bytes=[n*np.dtype(dt).itemsize for _,dt,n in INPUTS]
            self.outputs=[self._alloc(np.prod(s)*np.dtype(dt).itemsize) for s,dt in OUTPUT_SHAPES]
            self.output_bytes=[int(np.prod(s)*np.dtype(dt).itemsize) for s,dt in OUTPUT_SHAPES]
            self.heap=32<<20;self.heap_bytes=(32<<20)-4096
            # Scratch/staging each have the same exact verified bounds used by
            # the DSP build. Guard their unused following page.
            self.guards.extend((((80<<20)-4096,4096),((88<<20),4096),((64<<20)-4096,4096)))
            self.guards.append((SIZE-4096,4096))
            if self.preserve_vtcm:self.guards.extend((((96<<20)-128,128),(104<<20,128)))
            if self.l2_cache:
                # qd_l2_cache.h: worker-0 lease stack 0x07030000..0x07040000
                # (the engine checks its bottom 128 B), elementwise scratch
                # 0x07050000..0x07058000, conv near 0x07060000..0x07068000, metadata
                # from 0x07100000.
                # Threads: helper slots 0x07070000..0x07077e00, follower lease
                # stacks 0x07080000..0x070a0000.
                self.guards.extend(((112<<20,128),((112<<20)+0x40000,128),((112<<20)+0x58000,128),((112<<20)+0x68000,128),
                                    ((112<<20)+0x78000,128),((112<<20)+0xa0000,128)))
            for off,n in self.guards:self.mem[off:off+n]=b'\xa5'*n
            t=self.transport;t.arena[:32768]=bytes(32768)
            config=[MAGIC,1,image,len(self.image),*self.pools,*self.sizes,*self.inputs,*self.input_bytes,
                    *self.outputs,*self.output_bytes,self.heap,self.heap_bytes,len(code)]
            struct.pack_into('<33I',t.arena,4096,*config)
            if self.prepared_program is not None:
                extension=[0x31504e51,1,prepared,len(self.prepared_program),
                           *struct.unpack('<8I',hashlib.sha256(self.prepared_program).digest())]
                struct.pack_into('<12I',t.arena,4096+33*4,*extension)
            self.commands=self.header[42]
            return self
        except BaseException:
            self.close();raise

    def check_guards(self):
        for off,n in self.guards:
            if self.mem[off:off+n]!=b'\xa5'*n:raise RuntimeError(f'resident allocation guard changed at {off:#x}')

    def run(self,inputs,first=0,last=None,direct_hmx=None):
        direct=self.direct_hmx if direct_hmx is None else bool(direct_hmx)
        uploaded=time.monotonic()
        if last is None:last=self.commands
        if not 0<=first<=last<=self.commands:raise ValueError('invalid command range')
        for i,(name,dt,n) in enumerate(INPUTS):
            a=np.ascontiguousarray(inputs[i],dtype=dt).reshape(-1)
            if a.size!=n:raise ValueError(f'invalid {name} size')
            off=self.inputs[i];self.mem[off:off+a.nbytes]=a.tobytes()
        t=self.transport
        def prepare(seq):
            t.arena[128:176]=bytes(48)
            struct.pack_into('<7I',t.arena,0,0x32564451,6,seq-1,3,first,last,int(direct))
            struct.pack_into('<I',t.arena,3840,0)
            t.fence();struct.pack_into('<I',t.arena,8,seq);t.fence()
        start=time.monotonic();upload_ms=(start-uploaded)*1000
        result=t.submit(prepare,lambda:struct.unpack_from('<16I',t.arena,2048))
        self.seconds=time.monotonic()-start
        self.last={'direct_hmx':direct,'status':result[1],'command':result[2],'detail':result[3],
                   'dsp_ms':(result[4]+(result[5]<<32))/1843200,'calls':result[6],'hmx_calls':result[7],
                   'heap_used':result[8],'heap_peak':result[9],'scratch_peak':result[10],'l2_scratch_peak':result[11],'direct_hmx_calls':result[12],'seconds':self.seconds}
        if result[15]==0x52444951:
            self.last['native_direct_range']=True
            self.last['native_direct_trace']=list(struct.unpack_from('<16I',t.arena,1024))
            if self.last['native_direct_trace'][0]==0x31525051:
                self.last['native_prepared_program']=True
        if result[13]==0x31584e51:
            native=struct.unpack_from('<8Q',t.arena,2048+520)
            self.last['native_1x1']={'calls':native[0],
                'prepare_ms':native[1]/1843200,'leaf_issue_ms':native[2]/1843200,
                'drain_scatter_ms':native[3]/1843200,'unsupported':native[4],
                'failures':native[5],'workspace_bytes_staged':native[6],'reserved':native[7]}
        if result[14]==0x33584e51:
            native=struct.unpack_from('<8Q',t.arena,2048+592)
            self.last['native_3x3']={'calls':native[0],
                'prepare_ms':native[1]/1843200,'leaf_issue_ms':native[2]/1843200,
                'drain_scatter_ms':native[3]/1843200,'unsupported':native[4],
                'failures':native[5],'workspace_bytes_staged':native[6],'reserved':native[7]}
        self.last['upload_ms']=upload_ms
        cycles=struct.unpack_from('<6Q',t.arena,2048+64)
        self.last['prepare_ms']=cycles[0]/1843200;self.last['worker_ms']=cycles[1]/1843200
        self.last['hmx_phases_ms']=[n/1843200 for n in cycles[2:]]
        self.last['scratch_peak']=struct.unpack_from('<I',t.arena,2048+112)[0]
        diagnostics=struct.unpack_from('<8Q',t.arena,2048+128)
        self.last.update(zip(('read_ms','write_ms','pack_x_ms','prepare_weights_ms','upload_packet_ms','conv_weights_ms','quant_params_ms','conv_weights_miss_ms'),
                             (n/1843200 for n in diagnostics)))
        self.last['conv_weights_misses']=struct.unpack_from('<Q',t.arena,2048+192)[0]
        mc,self.last['malloc_calls'],self.last['malloc_steps']=struct.unpack_from('<3Q',t.arena,2048+200);self.last['malloc_ms']=mc/1843200
        phase_magic=struct.unpack_from('<I',t.arena,2048+288)[0]
        if phase_magic in (0x31504643,0x32504643,0x33504643,0x34504643,0x35504643,0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['conv_phases_ms']=dict(zip(('fi_u8','fi_u16','read_u8','read_u16','params_u8','params_u16','post_u8','post_u16'),
                                                (n/1843200 for n in struct.unpack_from('<8Q',t.arena,2048+224))))
        if phase_magic in (0x32504643,0x33504643,0x34504643,0x35504643,0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['metadata_ms']=struct.unpack_from('<Q',t.arena,2048+296)[0]/1843200
            self.last['dynamic_tcm_calls']=struct.unpack_from('<Q',t.arena,2048+304)[0]
        if phase_magic in (0x33504643,0x34504643,0x35504643,0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['fallback_phases_ms']=dict(zip(('prepare_u8','prepare_u16','merge_u8','merge_u16'),
                                                    (n/1843200 for n in struct.unpack_from('<4Q',t.arena,2048+312))))
            self.last['fallback_tcm_calls']=struct.unpack_from('<Q',t.arena,2048+344)[0]
        if phase_magic in (0x34504643,0x35504643,0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['fallback_l2_calls']=struct.unpack_from('<Q',t.arena,2048+352)[0]
        if phase_magic in (0x35504643,0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            calls,prep,apply=struct.unpack_from('<3Q',t.arena,2048+360)
            self.last.update(blocked_linearclip_calls=calls,blocked_linearclip_prepare_ms=prep/1843200,blocked_linearclip_apply_ms=apply/1843200)
        if phase_magic in (0x36504643,0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['blocked_linearclip_u8_u16_calls']=struct.unpack_from('<Q',t.arena,2048+384)[0]
        if phase_magic in (0x37504643,0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            calls,prep,apply=struct.unpack_from('<3Q',t.arena,2048+392)
            self.last.update(blocked_sign_calls=calls,blocked_sign_prepare_ms=prep/1843200,blocked_sign_apply_ms=apply/1843200)
        if phase_magic in (0x38504643,0x39504643,0x3a504643,0x3b504643,0x3c504643):
            calls,prep,apply=struct.unpack_from('<3Q',t.arena,2048+416)
            self.last.update(blocked_add_calls=calls,blocked_add_prepare_ms=prep/1843200,blocked_add_apply_ms=apply/1843200)
        if phase_magic in (0x39504643,0x3a504643,0x3b504643,0x3c504643):
            self.last['direct_input_pad_calls']=struct.unpack_from('<Q',t.arena,2048+440)[0]
        if phase_magic in (0x3a504643,0x3b504643,0x3c504643):
            self.last['direct_im2col_calls']=struct.unpack_from('<Q',t.arena,2048+448)[0]
        if phase_magic in (0x3b504643,0x3c504643):
            self.last['direct_space_to_depth_calls']=struct.unpack_from('<Q',t.arena,2048+456)[0]
        if phase_magic==0x3c504643:
            values=struct.unpack_from('<6Q',t.arena,2048+464)
            self.last['space_to_depth_phases_ms']=dict(zip(('prepare','apply','proof','gather','scatter'),(v/1843200 for v in values[:5])))
            self.last['space_to_depth_prepare_misses']=values[5]
        if result[0]!=MAGIC or result[1]:raise RuntimeError(f'resident graph failed: {self.last}')
        raw=np.frombuffer(bytes(t.arena[16384:16384+1280]),'<u8').reshape(2,80)
        self.profile=[{'variant':i,'ms':int(raw[0,i])/1843200,'count':int(raw[1,i])} for i in range(80) if raw[1,i]]
        self.handler_profile,self.executor_profile=self._read_handler_profile(t.arena)
        self.parallel_profile=self._read_parallel_profile(t.arena)
        self.check_guards()
        downloaded=time.monotonic()
        outs=[np.frombuffer(self.mem,dt,int(np.prod(shape)),off).copy().reshape(shape)
              for (shape,dt),off in zip(OUTPUT_SHAPES,self.outputs)]
        self.last['download_ms']=(time.monotonic()-downloaded)*1000
        self.last['total_seconds']=self.seconds+(upload_ms+self.last['download_ms'])/1000
        return outs

    @staticmethod
    def _read_parallel_profile(arena):
        """Thread record of a QDS_PREPARED_THREADS=3 engine (arena 24640), else {}.
        parts_run counts all executed parts; started/finished are per follower and
        must be equal after a run (a worker that never left its part shows as started>finished)."""
        r=struct.unpack_from('<16Q',arena,24640)
        if r[0]!=0x100000000|0x54505451:return {}
        k=('conv','ew','dm')
        return {'max_parts':int(r[1]&0xffffffff),'fault':int(r[1]>>32),'not_ready':int(r[2]),'declined':int(r[3]),
                'jobs_parallel':dict(zip(k,map(int,r[4:7]))),'jobs_serial':dict(zip(k,map(int,r[7:10]))),
                'parts_run':dict(zip(k,map(int,r[10:13]))),
                'started':[int(r[13]&0xffffffff),int(r[13]>>32)],'finished':[int(r[14]&0xffffffff),int(r[14]>>32)]}

    @staticmethod
    def _read_handler_profile(arena):
        """Prepared executor profile written by the engine (QDS_PREPARED_PROFILE=1):
        per-(opcode,variant) run ms/count/validator ms and the executor overhead
        buckets, all in ms (pcycles / 1843200). Empty when the engine has none."""
        head=struct.unpack_from('<8Q',arena,20480)
        if head[0]!=0x100000000|0x51444850:return [],{}
        ms=lambda v:v/1843200
        slots=int(head[7])
        rows=[]
        for i in range(slots):
            key,_,count,run,val=struct.unpack_from('<IIQQQ',arena,20480+64+i*32)
            if not key&0x10000:continue
            rows.append({'opcode':key&255,'variant':(key>>8)&255,'no_effect':bool(key>>24),
                         'count':int(count),'run_ms':ms(run),'validate_ms':ms(val)})
        rows.sort(key=lambda r:-(r['run_ms']+r['validate_ms']))
        return rows,{'preflight_ms':ms(head[1]),'validate_ms':ms(head[2]),'dispatch_ms':ms(head[3]),
                     'run_ms':ms(head[4]),'preflight_calls':int(head[5]&0xffffffff),'preflights_reused':int(head[5]>>32),
                     'executed_commands':int(head[6]&0xffffffff),'validated_commands':int(head[6]>>32)}

    def cancel(self):
        """Request cancellation at the next graph-command boundary."""
        struct.pack_into('<I',self.transport.arena,3840,1);self.transport.fence()

    def read_pool(self,p):
        return np.frombuffer(self.mem,np.uint8,self.sizes[p],self.pools[p]).copy()

    def close(self):
        try:
            if self.transport:
                started=self.transport.persistent or self.transport.job_active or bool(self.transport.calls)
                try:self.transport.stop_persistent()
                except BaseException:
                    if self.preserve_vtcm:self.transport.retain_session('VTCM lease exit did not complete')
                    raise
                if self.transport.arena is not None and struct.unpack_from('<I',self.transport.arena,3072)[0]:
                    self.transport.retain_session('resident JTLB/backup still active')
                    raise RuntimeError('resident JTLB still active; preserve allocation and recover before unloading')
                if self.preserve_vtcm and started and self.transport.arena is not None:
                    magic,phase,error=struct.unpack_from('<3I',self.transport.arena,3328)
                    save,restore=struct.unpack_from('<2Q',self.transport.arena,3456)
                    self.vtcm_lease={'magic':magic,'phase':phase,'error':error,'save_ms':save/1843200,'restore_ms':restore/1843200}
                    if magic!=0x31545651 or phase!=2 or error:
                        self.transport.retain_session(f'VTCM restoration unverified: {self.vtcm_lease}')
                        raise RuntimeError('VTCM lease restoration was not verified')
                    # Independently check the target's position-dependent
                    # fingerprints against the actual retained DDR snapshot.
                    raw=bytes(self.mem[96<<20:104<<20])
                    words=np.frombuffer(raw,'<u4').reshape(-1,32)
                    hx=int(np.bitwise_xor.reduce(words.reshape(-1)))
                    hs=int((words.sum(axis=1,dtype=np.uint64)*np.arange(1,len(words)+1,dtype=np.uint64)).sum(dtype=np.uint64))&0xffffffff
                    target=struct.unpack_from('<2I',self.transport.arena,3328+44)
                    self.vtcm_lease.update(backup_sha256=hashlib.sha256(raw).hexdigest(),host_fingerprint_equal=(hx,hs)==target)
                    if (hx,hs)!=target:
                        self.transport.retain_session('VTCM snapshot fingerprint differs from host calculation')
                        raise RuntimeError('VTCM snapshot fingerprint mismatch')
            self.exit_errors=[struct.unpack_from('<I',self.transport.arena,128+16*i+4)[0] for i in range(3)] if self.transport and self.transport.arena is not None else []
            if any(self.exit_errors):raise RuntimeError(f'resident bootstrap/stack check failed: {self.exit_errors}')
            if self.mem:
                self.check_guards()
                if hasattr(self,'pools') and self.mem[self.pools[5]:self.pools[5]+len(self.constants)]!=self.constants:
                    raise RuntimeError('resident constant pool changed')
        finally:
            try:
                if self.transport:self.transport.__exit__(None,None,None);self.transport=None
            finally:
                if self.mem:self.mem.close();self.mem=None

    def __exit__(self,*exc):self.close()
