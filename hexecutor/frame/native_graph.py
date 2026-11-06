"""Whole-graph native executor (qd_graph.c) over a QDRX image.

Python only loads the image and constants, owns the pools and, for the DSP
backend, wakes/releases the persistent workers through FrameTransport. Every
graph operation, and the DSP convolution protocol itself, runs in C.
"""
import ctypes
import hashlib
from pathlib import Path
import signal
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POOL_NAMES = {1: 'vtcm', 3: 'ddr3', 4: 'ddr4', 5: 'const', 6: 'spill'}
INPUTS = (('x', np.uint8, 81920), ('y', np.uint8, 81920), ('hidden', np.float32, 163840), ('t', np.float32, 32))
OUTPUT_SHAPES = (((1, 2, 320, 256), np.uint8), ((1, 2, 320, 256), np.float32))
ERRORS = {1: 'image', 2: 'tensor', 3: 'shape', 4: 'arguments', 5: 'range', 6: 'unsupported',
          7: 'memory', 8: 'DSP status', 9: 'DSP timeout', 10: 'DSP wake', 11: 'callback'}


class Stats(ctypes.Structure):
    _fields_ = [('calls', ctypes.c_uint64), ('hmx_calls', ctypes.c_uint64), ('bytes_uploaded', ctypes.c_uint64),
                ('hmx_phase_cycles', ctypes.c_uint64 * 4),
                ('prepare_ns', ctypes.c_uint64), ('wait_ns', ctypes.c_uint64), ('collect_ns', ctypes.c_uint64),
                ('conv_ns', ctypes.c_uint64), ('total_ns', ctypes.c_uint64), ('zero_ns', ctypes.c_uint64),
                ('variant_ns', ctypes.c_uint64 * 80), ('variant_count', ctypes.c_uint64 * 80),
                ('sequence', ctypes.c_uint32), ('persistent', ctypes.c_uint32),
                ('error_command', ctypes.c_uint32), ('error_detail', ctypes.c_uint32),
                ('bytes_x', ctypes.c_uint64 * 2), ('bytes_w', ctypes.c_uint64 * 2), ('bytes_out', ctypes.c_uint64 * 2),
                ('weights_reused', ctypes.c_uint64), ('hvx_phase_cycles', (ctypes.c_uint64 * 3) * 2),
                ('hmx_block_calls', ctypes.c_uint64)]


WAKE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)


def library():
    sources = [Path(__file__).with_name(n) for n in ('qd_graph.c', 'qd_resident.c', 'qd_graph.h', 'qd_resident.h', 'qd_resident_schema.h')]
    lib = Path(__file__).with_name('build') / 'libqd_graph.so'
    if not lib.exists() or lib.stat().st_mtime < max(s.stat().st_mtime for s in sources):
        lib.parent.mkdir(parents=True, exist_ok=True)
        flags = ['-O3', '-std=c11', '-fPIC', '-shared', '-ffp-contract=off', '-fno-fast-math', '-fno-strict-aliasing', '-Wall', '-Wextra']
        subprocess.run(['cc', *flags, str(sources[0]), str(sources[1]), '-lm', '-pthread', '-o', str(lib)], check=True)
    native = ctypes.CDLL(str(lib))
    vp, u32, u64p = ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)
    native.qdg_create.argtypes = [vp, u32, ctypes.POINTER(vp), u64p, ctypes.POINTER(ctypes.c_int)]
    native.qdg_create.restype = vp
    native.qdg_destroy.argtypes = [vp]
    native.qdg_set_dsp.argtypes = [vp, vp, vp, u32, u32, u32, u32, u32, ctypes.c_double, WAKE]
    native.qdg_set_cpu.argtypes = [vp]
    native.qdg_set_profile.argtypes = [vp, ctypes.c_int]
    native.qdg_set_hoist.argtypes = [vp, ctypes.c_int]
    native.qdg_set_native_hmx.argtypes = [vp, ctypes.c_int]
    native.qdg_set_block_output.argtypes = [vp, ctypes.c_int]
    native.qdg_set_helpers.argtypes = [vp, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    native.qdg_set_helpers.restype = ctypes.c_int
    native.qdg_run.argtypes = [vp, ctypes.POINTER(vp), u64p, ctypes.POINTER(vp), u64p, u32, u32]
    native.qdg_run.restype = ctypes.c_int
    native.qdg_commands.argtypes = [vp]
    native.qdg_commands.restype = ctypes.c_int
    native.qdg_stats.argtypes = [vp]
    native.qdg_stats.restype = ctypes.POINTER(Stats)
    native.qdg_reset_stats.argtypes = [vp]
    return native


class NativeGraph:
    def __init__(self, image, constants, recovered=None):
        self.native = library()
        self.image = Path(image).read_bytes()
        header = struct.unpack_from('<64I', self.image)
        if header[0] != int.from_bytes(b'QDRX', 'little'):
            raise ValueError('not a QDRX image')
        constants = Path(constants).read_bytes()
        if hashlib.sha256(constants).digest() != struct.pack('<8I', *header[16:24]):
            raise ValueError('constant pool does not match the QDRX image')
        sizes = dict(zip(range(7), header[24:31]))
        self.pools = {1: np.zeros(max(sizes[1], 8 << 20), np.uint8), 3: np.zeros(sizes[3], np.uint8),
                      4: np.zeros(sizes[4], np.uint8), 5: np.frombuffer(constants, np.uint8).copy(),
                      6: np.zeros(sizes[6], np.uint8)}
        ptrs = (ctypes.c_void_p * 7)(*[self.pools[p].ctypes.data if p in self.pools else None for p in range(7)])
        nbytes = (ctypes.c_uint64 * 7)(*[self.pools[p].nbytes if p in self.pools else 0 for p in range(7)])
        error = ctypes.c_int()
        self.handle = self.native.qdg_create(self.image, len(self.image), ptrs, nbytes, ctypes.byref(error))
        if not self.handle:
            raise ValueError(f'native graph rejected the image: {ERRORS.get(error.value, error.value)}')
        self.commands = self.native.qdg_commands(self.handle)
        self.outputs = [np.zeros(shape, dt) for shape, dt in OUTPUT_SHAPES]
        self._wake = None
        self.transport = None
        self._wake_error = None

    def close(self):
        if self.handle:
            self.native.qdg_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def use_cpu(self):
        self.transport = None
        self.native.qdg_set_cpu(self.handle)

    def use_dsp(self, transport, timeout=10.0):
        """Drive the transport's persistent payload from C. The transport must be open."""
        from frame_transport import CODE_BASE, CTRL, WORK
        self.transport = transport
        mem = ctypes.addressof(ctypes.c_uint8.from_buffer(transport.mem))

        def wake(phase):
            try:
                t = transport
                if phase == 0:
                    t.wait(lambda: t.get(CTRL) in (1, 3))
                    t._release_workers()
                else:
                    t._dispatch_critical = True
                    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                    try:
                        t.put(CTRL + 4, 1)
                        t.put(CTRL + 8, 4)
                        t.job_active = t.persistent = True
                        t.socket.sendall(b'work')
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
                        t._dispatch_critical = False
                return 0
            except BaseException as error:  # reported after the native call returns
                self._wake_error = error
                return 1
        self._wake = WAKE(wake)
        self.native.qdg_set_dsp(self.handle, transport.arena_address, mem + WORK + 24, CODE_BASE,
                                transport.sequence, int(transport.persistent), int(transport.raise_clock),
                                int(transport.use_hmx), timeout, self._wake)

    def set_native_hmx(self,enable):
        """0: exact; 1: native conversion; 2: native crouton load/store. Modes 1/2 change rounding."""
        self.native.qdg_set_native_hmx(self.handle,int(enable))

    def set_helpers(self, cpus):
        """Pin one arena-copy helper thread to each listed core (empty disables them)."""
        arr = (ctypes.c_int * max(1, len(cpus)))(*cpus)
        return self.native.qdg_set_helpers(self.handle, len(cpus), arr)

    def set_hoist(self, enable):
        self.native.qdg_set_hoist(self.handle, int(enable))

    def set_block_output(self, enable):
        """Opt into experimental direct HMX crouton output (default off)."""
        self.native.qdg_set_block_output(self.handle, int(enable))

    def set_profile(self, enable):
        self.native.qdg_set_profile(self.handle, int(enable))

    def stats(self):
        return self.native.qdg_stats(self.handle).contents

    def reset_stats(self):
        self.native.qdg_reset_stats(self.handle)

    def run(self, inputs, first=0, last=None):
        arrays = []
        for i, (name, dt, n) in enumerate(INPUTS):
            a = np.ascontiguousarray(inputs[i], dtype=dt).reshape(-1)
            if a.size != n:
                raise ValueError(f'input {name} has {a.size} values, expected {n}')
            arrays.append(a)
        ip = (ctypes.c_void_p * 4)(*[a.ctypes.data for a in arrays])
        ib = (ctypes.c_uint64 * 4)(*[a.nbytes for a in arrays])
        op = (ctypes.c_void_p * 2)(*[o.ctypes.data for o in self.outputs])
        ob = (ctypes.c_uint64 * 2)(*[o.nbytes for o in self.outputs])
        if self.transport is not None and self.transport.eye_pid is not None:
            self._check_eye()
        self._wake_error = None
        rc = self.native.qdg_run(self.handle, ip, ib, op, ob, first, self.commands if last is None else last)
        stats = self.stats()
        if self.transport is not None:
            self.transport.sequence = stats.sequence
            self.transport.persistent = bool(stats.persistent)
        if self._wake_error is not None:
            raise self._wake_error
        if rc:
            if rc == 9:
                raise TimeoutError('CDSP request timed out; workers may still own the arena')
            raise RuntimeError(f'native graph failed at command {stats.error_command}: '
                               f'{ERRORS.get(rc, rc)} (qd_graph.c line {stats.error_detail})')
        if self.transport is not None and self.transport.eye_pid is not None:
            self._check_eye()
        return self.outputs

    def _check_eye(self):
        import os
        t = self.transport
        state = os.pread(t.eye_stat_fd, 4096, 0).split(b') ', 1)[1][:1]
        if state != b'T':
            raise RuntimeError('paused eye client changed state during inference')
