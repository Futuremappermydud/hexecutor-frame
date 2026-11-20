"""CPU-only reference with native tensor access for model/kernel validation."""

# Direct CLI entry points use the same explicit package roots as the root task runner.
import sys as _path_sys
from pathlib import Path as _SourcePath
for _source_root in _SourcePath(__file__).resolve().parents:
    if (_source_root / "hexecutor/api.py").is_file():
        _path_sys.path[:0] = [str(_source_root), str(_source_root / "samples/depth-model/python"),
                             str(_source_root / "samples/depth-model/tools"),
                             str(_source_root / "hexecutor/benchmarks")]
        break
from hexecutor.paths import workspace as _workspace
PATHS = _workspace()

import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np
ROOT=(PATHS.repo)
sys.path.insert(0,str((PATHS.model_tools)))
import htp_emu


from hexecutor.offline.memory import native_library


class NativeMemory(htp_emu.Memory):
    def __init__(self,graph,const_pool,native):
        super().__init__(graph,const_pool)
        self.native=native
        self.descriptors={}
        self.plans={}
        self.pool_ptrs=np.zeros(max(p for p in self.pools if isinstance(p,int))+1,np.uintp)
        for p,buf in self.pools.items():
            if isinstance(p,int):self.pool_ptrs[p]=buf.ctypes.data

    def descriptor(self,tid):
        if tid not in self.descriptors:
            t=self.tensors[tid];shape=t['shape'];storage=t['storage']
            table=storage['blocks']
            bp=np.array([(v or {}).get('pool',0) if isinstance(v,dict) else 0 for v in table],np.uint32)
            bo=np.array([(v or {}).get('offset',0) if isinstance(v,dict) else 0 for v in table],np.uint32)
            layout=np.array([*shape['max_dimensions'][1:],storage.get('slice_offset',0),len(table)],np.int32)
            self.descriptors[tid]=(bp,bo,layout)
        return self.descriptors[tid]

    def transfer(self,tid,value,origin,store):
        key=(tid,value.shape,tuple(origin))
        if key not in self.plans:
            bp,bo,layout=self.descriptor(tid)
            dims=np.array(value.shape,np.int32);start=np.array(origin,np.int32)
            self.plans[key]=(dims,start,(self.pool_ptrs.ctypes.data,bp.ctypes.data,bo.ctypes.data,
                                        layout.ctypes.data,dims.ctypes.data,start.ctypes.data,value.dtype.itemsize))
        args=self.plans[key][2]
        self.native.qdm_transfer(value.ctypes.data,*args,store)

    def direct(self,tid):
        t=self.tensors[tid];d=t['storage']['direct'];dt=np.dtype(htp_emu.NP[t['dtype']])
        return np.ndarray(t['shape']['max_dimensions'],dtype=dt,buffer=self.pools[d['pool']],offset=d['offset'])

    def read(self,tid,full=False):
        t=self.tensors[tid];s=t['shape']
        dims=s['max_dimensions'] if full else s['dimensions'];pad=[0]*4 if full else s['padding']
        if t['storage'].get('direct'):
            return self.direct(tid)[tuple(slice(p,p+n) for p,n in zip(pad,dims))].copy()
        result=np.empty(dims,htp_emu.NP[t['dtype']]);self.transfer(tid,result,pad,0)
        return result

    def write(self,tid,value):
        t=self.tensors[tid];s=t['shape'];dims=list(s['dimensions'])
        if htp_emu.is_crouton(t):
            dims[3]=min(-(-(dims[3]+s['padding'][3])//32)*32,s['max_dimensions'][3])-s['padding'][3]
        value=np.asarray(value,dtype=htp_emu.NP[t['dtype']])
        if value.ndim==4 and value.shape[3]<dims[3]:
            value=np.pad(value,[(0,0)]*3+[(0,dims[3]-value.shape[3])])
        value=np.ascontiguousarray(np.broadcast_to(value,dims))
        if t['storage'].get('direct'):
            self.direct(tid)[tuple(slice(p,p+n) for p,n in zip(s['padding'],dims))]=value
        else:self.transfer(tid,value,s['padding'],1)

    def write_region(self,tid,value,origin=(0,0,0,0)):
        t=self.tensors[tid];value=np.ascontiguousarray(value,dtype=htp_emu.NP[t['dtype']])
        if t['storage'].get('direct'):
            self.direct(tid)[tuple(slice(p,p+n) for p,n in zip(origin,value.shape))]=value
        else:self.transfer(tid,value,origin,1)

    def unpack_weights(self,tid):
        t=self.tensors[tid];d=t['storage']['direct'];shape=t['shape']['max_dimensions']
        out=np.empty(shape,np.int8)
        self.native.qdm_unpack_weights(out.ctypes.data,self.pools[d['pool']].ctypes.data+d['offset'],
                                      *shape,int(t['class_name'] in ('Xpb','xpb')))
        return out


class NativeEmulator(htp_emu.Emulator):
    def __init__(self,graph,const_pool,inputs):
        super().__init__(graph,const_pool,inputs)
        self.native=native_library()
        self.mem=NativeMemory(graph,const_pool,self.native);self.t=self.mem.tensors
        self.weights={}
        # Only cache weights with an explicit immutable constant-pool producer.
        self.static_sources={}
        self.elementwise_args={}
        for node in graph['nodes']:
            if node['type_name'].startswith('ConvLayer.opt.weights_to_vtcm@'):
                src,dst=node['inputs'][0],node['outputs'][0]
                if self.t[src]['storage']['direct']['pool']==5:
                    if self.t[src]['class_name'] in ('Xwb','xwb','Xpb','xpb'):self.static_sources[dst]=src

    def run_node(self,node):
        fn=htp_emu.KERNELS[node['type_name']]
        if fn is htp_emu.k_hmx_conv:self.conv(node)
        elif fn in (htp_emu.k_add_u8,htp_emu.k_requant_u8):self.elementwise_u8(node,fn)
        else:fn(self,node)

    def elementwise_u8(self,node,fn):
        key=node['sequence']
        if key not in self.elementwise_args:
            self.elementwise_args[key]=htp_emu.call_args(self,node,'elementwise_add_crouton_u8' if fn is htp_emu.k_add_u8 else 'requantize_crouton_u8')
        regs,stack=self.elementwise_args[key]
        a=self.mem.read(node['inputs'][0]);result=np.empty_like(a)
        if fn is htp_emu.k_add_u8:
            b=self.mem.read(node['inputs'][1])
            if a.shape!=b.shape or stack[1]!=0:return fn(self,node)
            error=self.native.qdm_add_u8(result.ctypes.data,a.ctypes.data,b.ctypes.data,a.size,regs[4],regs[5],stack[0])
        else:
            if regs[4]!=0 or regs[5]!=0:return fn(self,node)
            error=self.native.qdm_requant_u8(result.ctypes.data,a.ctypes.data,a.size,regs[3])
        if error:return fn(self,node)
        self.mem.write(node['outputs'][0],result)

    def conv(self,node):
        xi,wi,fi=node['inputs'][:3];out=self.t[node['outputs'][0]]
        s2=node['type_name'].startswith('ConvLayer_s2')
        key=(wi,s2)
        w=self.weights.get(key)
        if w is None:
            w=self.mem.unpack_weights(self.static_sources.get(wi,wi))
            if s2:
                true=np.zeros_like(w)
                for k,(r,c) in enumerate(htp_emu.S2_TAPS):true[r,c]=w.reshape(9,*w.shape[2:])[k]
                w=htp_emu.s2_kernel(true)
            elif self.t[wi]['class_name']=='Xwb':w=w[:,::-1]
            w=np.ascontiguousarray(w,dtype=np.int8)
            if wi in self.static_sources:
                w.flags.writeable=False
                self.weights[key]=w
        mult,bias=htp_emu.read_fi(self.mem,fi)
        sixteen=self.t[xi]['class_name'] in ('CH','cH')
        oh,ow,co=out['shape']['dimensions'][1:];omax=out['shape']['max_dimensions']
        co=min(-(-co//32)*32,w.shape[3],len(mult));xs=self.t[xi]['shape']
        xf=self.mem.read(xi,full=True)[0]
        x=xf[xs['padding'][1]:,xs['padding'][2]:,:w.shape[2]]
        cw=4 if sixteen else 8
        oh=min(-(-oh//8)*8,omax[1],x.shape[0]-w.shape[0]+1)
        ow=min(-(-ow//cw)*cw,omax[2],x.shape[1]-w.shape[1]+1)
        acc=htp_emu.conv_accumulate(x[:oh+w.shape[0]-1,:ow+w.shape[1]-1],w[...,:co],htp_emu.q(self.t[xi])[1])
        acc=np.ascontiguousarray(acc,np.int64)
        result=np.empty(acc.shape,np.uint16 if sixteen else np.uint8)
        self.native.qdm_requant(result.ctypes.data,acc.ctypes.data,mult.ctypes.data,bias.ctypes.data,
                               oh*ow,co,sixteen)
        self.mem.write_region(out['id'],result[None],(0,*out['shape']['padding'][1:3],0))
