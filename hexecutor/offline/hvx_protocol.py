"""Versioned, bounded matrix packets shared by the local and Frame transports."""
import struct

import numpy as np

MAGIC = 0x31564451
VERSION = 1
ARENA_OFFSET = 0x1800000
ARENA_BYTES = 0x800000
STATUS_OFFSET = 128
CHUNK_K = 256
CODE_BASE = 0x87606000  # fixed loader page (qd_loader.S)
CODE_BYTES = 4096
PAYLOAD_BASE = 0x87610000  # rewritable kernel region, invalidated per call
PAYLOAD_BYTES = 0x8000
HEADER = struct.Struct('<16I')
STATUS = struct.Struct('<4I')


def align128(n):
    return (n + 127) & ~127


def packet_layout(m, n, k):
    if not (0 < m <= 8192 and 0 < n <= 1024 and 0 < k <= 8192 and n % 32 == 0):
        raise ValueError(f'unsupported matrix dimensions {(m, n, k)}')
    chunks = (k + CHUNK_K - 1) // CHUNK_K
    a_offset = 4096
    b_offset = align128(a_offset + m * k * 2)
    c_offset = align128(b_offset + k * n * 4)
    end = c_offset + chunks * m * n * 4
    if end > ARENA_BYTES:
        raise ValueError(f'matrix packet needs {end} bytes, arena has {ARENA_BYTES}')
    return a_offset, b_offset, c_offset, end, chunks


def pack(arena, a, b, sequence):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu':
        raise ValueError('matrix operands must be rank-2 integer arrays')
    if np.any(a < -32768) or np.any(a > 32767):
        raise ValueError('activations must fit signed halfwords')
    if np.any(b < -128) or np.any(b > 127):
        raise ValueError('weights must be signed 8-bit values')
    a = np.ascontiguousarray(a, dtype='<i2')
    b = np.ascontiguousarray(b, dtype='<i4')
    m, k = a.shape
    if b.ndim != 2 or b.shape[0] != k:
        raise ValueError('matrix dimensions do not match')
    n = b.shape[1]
    ao, bo, co, end, chunks = packet_layout(m, n, k)
    arena[ao:ao + a.nbytes] = a.tobytes()
    arena[bo:bo + b.nbytes] = b.tobytes()
    # Status is uncached on the DSP. Sequence is published only after all data.
    arena[STATUS_OFFSET:STATUS_OFFSET + 48] = bytes(48)
    HEADER.pack_into(arena, 0, MAGIC, VERSION, sequence, ARENA_BYTES, m, n, k, ao, bo, co, *([0] * 6))
    return m, n, k, co, chunks, a.nbytes + b.nbytes


def result(arena, spec, sequence):
    m, n, k, co, chunks, _ = spec
    for tid in range(3):
        done, error, rows, _ = STATUS.unpack_from(arena, STATUS_OFFSET + tid * STATUS.size)
        if done != sequence or error:
            raise RuntimeError(f'HVX worker {tid}: sequence={done}, error={error}')
        expected = len(range(tid, m, 3))
        if rows != expected:
            raise RuntimeError(f'HVX worker {tid}: completed {rows}/{expected} rows')
    values = np.frombuffer(arena, '<i4', chunks * m * n, co).copy().reshape(chunks, m, n)
    return values.sum(axis=0, dtype=np.int64)
