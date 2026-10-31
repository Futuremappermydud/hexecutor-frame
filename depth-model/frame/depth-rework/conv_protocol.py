"""Packets for the persistent HVX convolution payload (qd_conv.h)."""
import ctypes
from pathlib import Path
import struct
import subprocess

import numpy as np

from hvx_protocol import ARENA_BYTES, STATUS, STATUS_OFFSET

MAGIC = 0x32564451
VERSION = 2
OP_CONV, OP_EXIT = 1, 2
BARRIER_OFFSET = 64
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
    src = Path(__file__).with_name('qd_fence.c')
    if not lib.exists() or lib.stat().st_mtime < src.stat().st_mtime:
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cc', '-O2', '-shared', '-fPIC', str(src), '-o', str(lib)], check=True)
    fence = ctypes.CDLL(str(lib)).qd_fence
    fence.argtypes, fence.restype = (), None
    return fence


def conv_layout(x_shape, w_shape):
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
    xb, wb, cb = h * g * wp * 4, kh * kw * g * n * 4, oh * n * owp * 4
    if chunked(xb + 128) > VTCM_BYTES or chunked(wb) > L2TCM_BYTES:
        raise ValueError(f'convolution operands exceed TCM: {x_shape} * {w_shape}')
    xo = 4096
    wo = xo + chunked(xb + 128)
    co = wo + chunked(wb)
    if co + cb > ARENA_BYTES:
        raise ValueError('convolution packet exceeds the arena')
    return dict(h=h, g=g, wp=wp, kh=kh, kw=kw, n=n, oh=oh, ow=ow, owp=owp,
                xo=xo, wo=wo, co=co, xb=xb, wb=wb, cb=cb)


def pack_conv(arena, x, w, sequence, fence=None):
    """x: uint8 [H,W,C]; w: integer [kh,kw,C,N] within int8. Publishes the
    sequence only after all operands and the rest of the header are written."""
    if x.dtype != np.uint8:
        raise ValueError('convolution activations must be uint8')
    if np.any(w < -128) or np.any(w > 127):
        raise ValueError('weights must be signed 8-bit values')
    L = conv_layout(x.shape, w.shape)
    h, g, wp, kh, kw, n = L['h'], L['g'], L['wp'], L['kh'], L['kw'], L['n']
    xd = np.frombuffer(arena, np.uint8, chunked(L['xb'] + 128), L['xo'])
    xv = xd[:L['xb']].reshape(h, g, wp, 4)
    xd[:] = 0
    c = x.shape[2]
    xc = x if c == 4 * g else np.pad(x, ((0, 0), (0, 0), (0, 4 * g - c)))
    xv[:, :, :x.shape[1], :] = xc.reshape(h, x.shape[1], g, 4).transpose(0, 2, 1, 3)
    wd = np.frombuffer(arena, np.int8, L['wb'], L['wo']).reshape(kh, kw, g, n, 4)
    wc = w.astype(np.int8)
    if c != 4 * g:
        wc = np.pad(wc, ((0, 0), (0, 0), (0, 4 * g - c), (0, 0)))
    wd[:] = wc.reshape(kh, kw, g, 4, n).transpose(0, 1, 2, 4, 3)
    arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
    struct.pack_into('<I', arena, BARRIER_OFFSET, 0)
    fields = [MAGIC, VERSION, 0, OP_CONV, h, wp, g, kh, kw, n, L['oh'], L['owp'],
              L['xo'], L['wo'], L['co'], ARENA_BYTES, 0, 0, 0, 0]
    COMMAND.pack_into(arena, 0, *fields)
    if fence:
        fence()
    struct.pack_into('<I', arena, 8, sequence)
    if fence:
        fence()
    return L


def pack_exit(arena, sequence, fence=None):
    arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
    fields=[0]*11
    fields[5]=(~sequence)&0xffffffff
    COMMAND.pack_into(arena, 0, MAGIC, VERSION, sequence - 1, OP_EXIT, *fields, ARENA_BYTES, 0, 0, 0, 0)
    if fence:
        fence()
    struct.pack_into('<I', arena, 8, sequence)
    if fence:
        fence()


def conv_result(arena, L, sequence):
    units = L['oh'] * -(-L['owp'] // 64) * (L['n'] // NB)
    total = 0
    for tid in range(3):
        done, error, got, _ = STATUS.unpack_from(arena, STATUS_OFFSET + tid * STATUS.size)
        if done != sequence or error:
            raise RuntimeError(f'HVX worker {tid}: sequence={done}, error={error}')
        total += got
    if total != units:
        raise RuntimeError(f'HVX workers completed {total}/{units} units')
    out = np.frombuffer(arena, '<i4', L['cb'] // 4, L['co']).reshape(L['oh'], L['n'], L['owp'])
    return out[:, :, :L['ow']].transpose(0, 2, 1).copy()
