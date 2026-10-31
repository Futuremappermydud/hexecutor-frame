"""Local packet tests. These backends do not claim DSP/HVX execution."""
import ctypes
from pathlib import Path

import numpy as np

from conv_protocol import L2TCM_BYTES, VTCM_BYTES, conv_result, pack_conv
from hvx_protocol import ARENA_BYTES, CHUNK_K, STATUS_OFFSET, STATUS, pack, result


class LocalTransport:
    def __init__(self, library=None, conv_library=None):
        self.arena = bytearray(ARENA_BYTES)
        self.sequence = 0
        self.library = self.conv_library = None
        if library:
            self.library = ctypes.CDLL(str(Path(library).resolve()))
            self.library.qd_host_run.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            self.library.qd_host_run.restype = None
        if conv_library:
            self.conv_library = ctypes.CDLL(str(Path(conv_library).resolve()))
            self.conv_library.qdc_host_run.argtypes = (ctypes.c_void_p,) * 3 + (ctypes.c_uint32,) * 2
            self.conv_library.qdc_host_run.restype = None
            self.vtcm = bytearray(VTCM_BYTES)
            self.l2 = bytearray(L2TCM_BYTES)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def conv(self, x, w):
        """Exact sum(x*w) for uint8 x through the persistent-payload packet format."""
        self.sequence += 1
        L = pack_conv(self.arena, x, w, self.sequence)
        if self.conv_library:
            ptrs = [(ctypes.c_uint8 * len(b)).from_buffer(b) for b in (self.arena, self.vtcm, self.l2)]
            for phase in (0, 1):
                for tid in range(3):
                    self.conv_library.qdc_host_run(*ptrs, tid, phase)
        else:
            raise RuntimeError('LocalTransport.conv needs the native conv library')
        return conv_result(self.arena, L, self.sequence)

    def matmul(self, a, b):
        self.sequence += 1
        spec = pack(self.arena, a, b, self.sequence)
        m, n, k, co, chunks, _ = spec
        if self.library:
            buf = (ctypes.c_uint8 * len(self.arena)).from_buffer(self.arena)
            for tid in range(3):
                self.library.qd_host_run(buf, tid)
        else:
            for chunk in range(chunks):
                lo, hi = chunk * CHUNK_K, min((chunk + 1) * CHUNK_K, k)
                partial = a[:, lo:hi].astype(np.int64) @ b[lo:hi].astype(np.int64)
                if np.any(partial < -(2**31)) or np.any(partial >= 2**31):
                    raise OverflowError('DSP partial sum would overflow int32')
                off = co + chunk * m * n * 4
                self.arena[off:off + m * n * 4] = partial.astype('<i4').tobytes()
            for tid in range(3):
                STATUS.pack_into(self.arena, STATUS_OFFSET + tid * STATUS.size,
                                 self.sequence, 0, len(range(tid, m, 3)), 0)
        return result(self.arena, spec, self.sequence)
