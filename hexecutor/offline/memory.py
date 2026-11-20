# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native host tensor/packet primitives shared by execution backends."""
import ctypes
import subprocess
from hexecutor.paths import workspace
PATHS=workspace()

def native_library():
    source=(PATHS.native / 'qd_memory.c')
    lib=(PATHS.native / 'build/libqd_memory.so')
    if not lib.exists() or lib.stat().st_mtime<source.stat().st_mtime:
        lib.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['cc','-O3','-std=c11','-fPIC','-shared',str(source),'-o',str(lib)],check=True)
    native=ctypes.CDLL(str(lib))
    native.qdm_transfer.argtypes=[ctypes.c_void_p]*7+[ctypes.c_uint32]*2
    native.qdm_transfer.restype=None
    native.qdm_unpack_weights.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*5
    native.qdm_unpack_weights.restype=None
    native.qdm_requant.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint32]*3
    native.qdm_requant.restype=None
    native.qdm_pack_x.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*6
    native.qdm_pack_x.restype=None
    native.qdm_pack_w.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*5
    native.qdm_pack_w.restype=None
    native.qdm_collect.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*4
    native.qdm_collect.restype=None
    native.qdm_copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    native.qdm_copy.restype=None
    native.qdm_poll.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32]
    native.qdm_poll.restype=ctypes.c_int
    native.qdm_quant_params.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint32]*2+[ctypes.c_int32]
    native.qdm_quant_params.restype=ctypes.c_int
    native.qdm_collect_u8.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*4
    native.qdm_collect_u8.restype=None
    native.qdm_pack_w_hmx.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*6
    native.qdm_pack_w_hmx.restype=None
    native.qdm_collect_hmx_u8.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint32]*4
    native.qdm_collect_hmx_u8.restype=None
    native.qdm_add_u8.argtypes=[ctypes.c_void_p]*3+[ctypes.c_size_t]+[ctypes.c_int32]*3
    native.qdm_add_u8.restype=ctypes.c_int
    native.qdm_requant_u8.argtypes=[ctypes.c_void_p]*2+[ctypes.c_size_t,ctypes.c_uint32]
    native.qdm_requant_u8.restype=ctypes.c_int
    return native
