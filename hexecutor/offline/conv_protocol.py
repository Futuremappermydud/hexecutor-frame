"""Packets for the persistent HVX convolution payload (qd_conv.h)."""

from hexecutor.paths import workspace
PATHS = workspace()

import ctypes
from pathlib import Path
import struct
import subprocess

import numpy as np

from hexecutor.offline.hvx_protocol import ARENA_BYTES, STATUS, STATUS_OFFSET

MAGIC = 0x32564451
VERSION = 6
OP_CONV, OP_EXIT = 1, 2
BARRIER_OFFSET = 96
VTCM_BYTES = 0x2d0000
L2TCM_BYTES = 0x100000
CHUNK = 1024
NB = 8
COMMAND = struct.Struct('<20I')
IDLE_EXIT = 4
EXITED = 0x54495845


def chunked(n):
    return (n + CHUNK - 1) & ~(CHUNK - 1)


def fence_library(build_dir):
    """Compile (once) and load the host memory-barrier helper."""
    build_dir = Path(build_dir)
    lib = build_dir / 'libqd_fence.so'
    src = (PATHS.native / 'qd_fence.c')
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cc', '-O2', '-shared', '-fPIC', str(src), '-o', str(lib)], check=True)
    fence = ctypes.CDLL(str(lib)).qd_fence
    fence.argtypes, fence.restype = (), None
    return fence


def conv_layout(x_shape, w_shape, quantized=False, hmx=False):
    h, width, c = x_shape
    kh, kw, wc, n = w_shape
    if wc != c or n % NB or n > 1024 or kh > 8 or kw > 8:
        raise ValueError(f'unsupported convolution {x_shape} * {w_shape}')
    g = (c + 3) // 4
    oh, ow = h - kh + 1, width - kw + 1
    owp = -(-ow // 32) * 32
    wp = owp + kw - 1
    if oh <= 0 or ow <= 0 or oh > 64 or owp > 2048 or g > 256:
        raise ValueError(f'unsupported convolution {x_shape} * {w_shape}')
    xb, wb, cb = h * g * wp * 4, kh * kw * g * n * 4, oh * n * owp * (1 if quantized else 4)
    kb=(kh*kw*g+7)//8
    if hmx:
        if not quantized or n%32:raise ValueError('HMX requires u8 quantization and 32-channel output groups')
        wb=kb*(n//32)*1024;cb=oh*(owp//32)*(n//32)*1024
        xa=(xb+128+2047)&~2047
        b=xa+oh*(owp//32)*kb*2048
        c=(b+wb+2047)&~2047
        end=((c+3*cb+((n*8+127)&~127)+256+2047)&~2047)+2048
        if end>VTCM_BYTES:raise ValueError('HMX packed tensors exceed the scratch VTCM')
    if chunked(xb + 128) > VTCM_BYTES or chunked(wb) > L2TCM_BYTES:
        raise ValueError(f'convolution operands exceed TCM: {x_shape} * {w_shape}')
    pb=chunked(n*12) if quantized else 0
    xo = 4096+pb
    wo = xo + chunked(xb + 128)
    co = wo + chunked(wb)
    if co + cb > ARENA_BYTES:
        raise ValueError('convolution packet exceeds the arena')
    return dict(h=h, g=g, wp=wp, kh=kh, kw=kw, n=n, oh=oh, ow=ow, owp=owp,
                xo=xo, wo=wo, co=co, xb=xb, wb=wb, cb=cb, pb=pb, quantized=quantized, hmx=hmx, kb=kb)


def pack_conv(arena, x, w, sequence, fence=None, raise_clock=True, native=None, weight_cache=None, quantization=None, hmx=False):
    """x: uint8 [H,W,C]; w: integer [kh,kw,C,N] within int8. Publishes the
    sequence only after all operands and the rest of the header are written."""
    if x.dtype != np.uint8:
        raise ValueError('convolution activations must be uint8')
    if w.dtype != np.int8 and (np.any(w < -128) or np.any(w > 127)):
        raise ValueError('weights must be signed 8-bit values')
    L = conv_layout(x.shape, w.shape, quantization is not None, hmx)
    h, g, wp, kh, kw, n = L['h'], L['g'], L['wp'], L['kh'], L['kw'], L['n']
    if native is None:
        xd = np.frombuffer(arena, np.uint8, chunked(L['xb'] + 128), L['xo'])
        xv = xd[:L['xb']].reshape(h, g, wp, 4)
        xd[:] = 0
        c = x.shape[2]
        xc = x if c == 4 * g else np.pad(x, ((0, 0), (0, 0), (0, 4 * g - c)))
        xv[:, :, :x.shape[1], :] = xc.reshape(h, x.shape[1], g, 4).transpose(0, 2, 1, 3)
        wc=w.astype(np.int8)
        if c!=4*g:wc=np.pad(wc,((0,0),(0,0),(0,4*g-c),(0,0)))
        wd=np.frombuffer(arena,np.int8,L['wb'],L['wo'])
        if hmx:
            flat=wc.reshape(-1,n)
            flat=np.pad(flat,((0,L['kb']*32-len(flat)),(0,0)))
            wd[:]=flat.reshape(L['kb'],8,4,n//32,32).transpose(3,0,1,4,2).reshape(-1)
        else:wd[:]=wc.reshape(kh,kw,g,4,n).transpose(0,1,2,4,3).reshape(-1)
    else:
        if not x.flags.c_contiguous:
            x=np.ascontiguousarray(x)
        raw=np.frombuffer(arena,np.uint8)
        native.qdm_pack_x(raw.ctypes.data+L['xo'],x.ctypes.data,h,x.shape[1],x.shape[2],g,wp,chunked(L['xb']+128))
        key=(w.ctypes.data,w.shape,hmx)
        cached=weight_cache.get(key) if weight_cache is not None and not w.flags.writeable else None
        if cached is None:
            packed=np.empty(L['wb'],np.int8)
            wc=np.ascontiguousarray(w,dtype=np.int8)
            if hmx:native.qdm_pack_w_hmx(packed.ctypes.data,wc.ctypes.data,kh,kw,x.shape[2],n,g,L['kb'])
            else:native.qdm_pack_w(packed.ctypes.data,wc.ctypes.data,kh,kw,x.shape[2],n,g)
            if weight_cache is not None and not w.flags.writeable:
                weight_cache[key]=(w,packed)
        else:
            _,packed=cached
        native.qdm_copy(raw.ctypes.data+L['wo'],packed.ctypes.data,L['wb'])
    if quantization is not None:
        quantization=np.ascontiguousarray(quantization,dtype=np.int32)
        if quantization.shape != (n,3):raise ValueError('quantization must be [N,3] int32')
        arena[4096:4096+L['pb']]=bytes(L['pb'])
        arena[4096:4096+quantization.nbytes]=quantization.tobytes()
        L['quantization']=quantization
    arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
    fields = [MAGIC, VERSION, 0, OP_CONV, h, wp, g, kh, kw, n, L['oh'], L['owp'],
              L['xo'], L['wo'], L['co'], ARENA_BYTES, 20000, (0 if raise_clock else 1) | (2 if quantization is not None else 0) | (4 if hmx else 0),
              4096 if quantization is not None else 0, 0]
    publish_header(arena, fields, sequence, fence)
    return L


def publish_header(arena, fields, sequence, fence=None):
    # Never overwrite the live sequence while publishing the other words.
    # Workers retain the previous command until the final aligned u32 store.
    header = COMMAND.pack(*fields)
    arena[:8] = header[:8]
    arena[12:COMMAND.size] = header[12:]
    if fence:
        fence()
    struct.pack_into('<I', arena, 8, sequence)
    if fence:
        fence()


def pack_exit(arena, sequence, fence=None):
    arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
    fields = [MAGIC, VERSION, 0, OP_EXIT] + [0] * 11 + [ARENA_BYTES, 20000, 0, 0, 0]
    fields[9] = (~sequence) & 0xffffffff  # Qwen's existing EXIT integrity word
    publish_header(arena, fields, sequence, fence)


def conv_result(arena, L, sequence, native=None):
    units = L['oh'] * -(-L['owp'] // 64) * (L['n'] // NB)
    total = 0
    for tid in range(3):
        done, error, got, _ = STATUS.unpack_from(arena, STATUS_OFFSET + tid * STATUS.size)
        if done != sequence or error:
            raise RuntimeError(f'HVX worker {tid}: sequence={done}, error={error}')
        total += got
    if total != units:
        raise RuntimeError(f'HVX workers completed {total}/{units} units')
    if native is not None:
        out=np.empty((L['oh'],L['ow'],L['n']),np.uint8 if L['quantized'] else np.int64)
        raw=np.frombuffer(arena,np.uint8)
        staging=np.empty(L['cb'],np.uint8)
        native.qdm_copy(staging.ctypes.data,raw.ctypes.data+L['co'],L['cb'])
        if L['hmx']:
            native.qdm_collect_hmx_u8(out.ctypes.data,staging.ctypes.data,L['oh'],L['ow'],L['n'],L['owp'])
        else:
            collect=native.qdm_collect_u8 if L['quantized'] else native.qdm_collect
            collect(out.ctypes.data,staging.ctypes.data,L['oh'],L['ow'],L['n'],L['owp'])
        return out
    if L['hmx']:
        out=np.frombuffer(arena,np.uint8,L['cb'],L['co']).reshape(L['oh']*L['owp']//32,L['n']//32,32,32)
        return out.transpose(0,2,1,3).reshape(L['oh'],L['owp'],L['n'])[:,:L['ow']].copy()
    if L['quantized']:
        out=np.frombuffer(arena,np.uint8,L['cb'],L['co']).reshape(L['oh'],L['n']//4,L['owp']//32,4,32)
        return out.transpose(0,2,4,1,3).reshape(L['oh'],L['owp'],L['n'])[:,:L['ow']].copy()
    out = np.frombuffer(arena, '<i4', L['cb'] // 4, L['co']).reshape(L['oh'], L['n'], L['owp'])
    return out[:, :, :L['ow']].transpose(0, 2, 1).copy()
