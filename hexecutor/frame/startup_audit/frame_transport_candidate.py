"""Bounded Frame CDSP transport using the existing Valve worker callback.

Uses the established FLOPS wake-up path, so eye inference is paused for the
session and restored afterwards. No daemon stop, firmware replacement, QuRT
or FastRPC. This module accesses the device only when FrameTransport is entered.
"""
import array
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import select
import struct
import subprocess
import time

from conv_protocol import EXITED, conv_layout, conv_result, fence_library, pack_conv, pack_exit
from hvx_protocol import (ARENA_BYTES, ARENA_OFFSET, CODE_BASE, CODE_BYTES,
                          PAYLOAD_BASE, PAYLOAD_BYTES, STATUS, STATUS_OFFSET,
                          pack, result)

FIRMWARE = Path('/opt/steamvr/drivers/cv/bin/linuxarm64/cdsp_sm8650.mbn')
FIRMWARE_SHA256 = '4f9fd567eb582c24f1f960a6525fafb0bd75733223ffa7ce8d6541cfd2a62b1c'
CTRL = 0xA91000
WORK = 0xB12000
DMA_BYTES = 0x2000000


def _process_state(pid):
    """A process may exit between pgrep/exists and reading its proc file."""
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1][0]
    except (FileNotFoundError,ProcessLookupError):
        return None


def _wait_for_single_eye(timeout=2.0):
    """Allow SteamVR's replacement client to settle without starting another."""
    deadline=time.monotonic()+timeout
    while True:
        found=subprocess.run(['pgrep','-x','eyetracking'],capture_output=True,text=True)
        live=[int(p) for p in found.stdout.split() if _process_state(p) not in (None,'Z')]
        if len(live)==1:return live[0]
        remaining=deadline-time.monotonic()
        if remaining<=0:return None
        time.sleep(min(0.1,remaining))


def load_callback(path, base=CODE_BASE, size=CODE_BYTES):
    elf = Path(path).read_bytes()
    if elf[:7] != b'\x7fELF\x01\x01\x01' or struct.unpack_from('<H', elf, 18)[0] != 164:
        raise ValueError('callback must be a little-endian 32-bit Hexagon ELF')
    entry, phoff = struct.unpack_from('<II', elf, 24)
    phsize, count = struct.unpack_from('<HH', elf, 42)
    if entry != base or phsize != 32:
        raise ValueError(f'callback must enter the dedicated region at {base:#x}')
    page = bytearray(size)
    loads = 0
    for i in range(count):
        typ, offset, va, _, fs, ms, flags, _ = struct.unpack_from('<8I', elf, phoff + i * phsize)
        if typ != 1:
            continue
        if not (base <= va <= base + size and fs <= ms
                and ms <= base + size - va and offset + fs <= len(elf)):
            raise ValueError('callback load segment extends outside its code region')
        page[va - base:va - base + fs] = elf[offset:offset + fs]
        loads += 1
    if not loads:
        raise ValueError('callback has no loadable segment')
    return bytes(page), hashlib.sha256(elf).hexdigest()


class FrameTransport:
    """`callback` is the fixed loader page; `payload` is the kernel it invalidates
    from the I-cache and jumps to, so payloads may change between sessions."""
    def __init__(self, callback, payload, timeout=120, raise_clock=True, use_hmx=False):
        self.code, self.callback_sha256 = load_callback(callback)
        self.payload, self.payload_sha256 = load_callback(payload, PAYLOAD_BASE, PAYLOAD_BYTES)
        self.timeout = timeout
        self.raise_clock = raise_clock
        self.use_hmx=use_hmx
        self.hmx_calls=0
        self.hmx_phase_cycles=[0]*4
        from fast_depth import native_library
        self.packet_native = native_library()
        self.weight_cache = {}
        self.hmx_layout_cache = {}
        self.sequence = 0
        self.mem = self.arena = self.socket = self.lock = None
        self.fds = []
        self.saved_work = self.saved_code = self.saved_payload = None
        self.original_eye = None
        self.eye_pid = None
        self.eye_stat_fd = -1
        self.eye_stopped = False
        self.job_active = False
        # `installed` means the code/payload pages are loaded. `registered`
        # means WORK+24 points at this callback and external dispatch can enter.
        self.installed = False
        self.registered = False
        self.fence = fence_library(Path(__file__).resolve().parent / 'build')
        self.persistent = False
        self.calls = self.bytes_uploaded = 0
        self.dispatch_seconds = self.prepare_seconds = self.wait_seconds = self.collect_seconds = 0.0
        self._dispatch_critical = self._closing = self._cancel_requested = False
        self._stuck = False
        self._retain_reason = None
        self._saved_signals = {}

    def get(self, offset):
        return struct.unpack_from('<I', self.mem, offset)[0]

    def retain_session(self, reason):
        """Keep callback data and the paused eye client for explicit recovery.

        A failed memory restore can be discovered after workers have exited.
        Retention must therefore be independent of job_active/persistent.
        """
        self._retain_reason = str(reason)
        self._stuck = True

    def put(self, offset, value):
        struct.pack_into('<I', self.mem, offset, value)

    def _control(self):
        return (self.get(CTRL), self.get(CTRL + 4), self.get(CTRL + 8))

    def _eye_is_paused(self):
        if self.eye_pid is None:
            return True
        state = os.pread(self.eye_stat_fd, 4096, 0).split(b') ', 1)[1][:1]
        return state == b'T'

    def _register_callback(self):
        """Publish the callback only after the first command is prepared."""
        if self.registered:
            if self.get(WORK + 24) != CODE_BASE:
                self.retain_session('registered callback pointer changed before dispatch')
                raise RuntimeError('registered callback pointer changed before dispatch')
            if self._control() != (1, 0, 1):
                raise RuntimeError('DSP is not in the verified idle state before dispatch')
            if not self._eye_is_paused():
                raise RuntimeError('paused eye client changed state before dispatch')
            return
        if not self.installed or self.mem is None:
            raise RuntimeError('callback pages are not loaded')
        if self.job_active or self._control() != (1, 0, 1):
            raise RuntimeError('DSP is not in the verified idle state before callback registration')
        if self.get(WORK + 24) != 0:
            raise RuntimeError('another DSP worker callback is installed')
        if not self._eye_is_paused():
            raise RuntimeError('paused eye client changed state before callback registration')
        code_offset = CODE_BASE - 0x87000000
        if self.mem[code_offset:code_offset + CODE_BYTES] != self.code:
            raise RuntimeError('loaded callback page changed before registration')
        self.put(WORK + 24, CODE_BASE)
        self.registered = True

    def wait(self, predicate, timeout=None, spin=0.0):
        """Poll `predicate`; busy-poll for the first `spin` seconds, then sleep."""
        now = time.monotonic()
        end = now + (self.timeout if timeout is None else timeout)
        busy = now + spin
        while not predicate():
            if self.installed and not self._closing and self.get(WORK + 24) != CODE_BASE:
                raise RuntimeError('DSP callback disappeared after a reset/suspend; keep the Frame awake during inference')
            if self.socket is not None:
                while select.select([self.socket], [], [], 0)[0]:
                    if not self.socket.recv(4096):
                        raise RuntimeError('dsp_service disconnected during the request')
            if time.monotonic() >= end:
                raise TimeoutError('CDSP request timed out; workers may still own the arena')
            if time.monotonic() >= busy:
                time.sleep(0.0002)

    def workers_done(self):
        return all(STATUS.unpack_from(self.arena, STATUS_OFFSET + i * STATUS.size)[0] == self.sequence
                   for i in range(3))

    def __enter__(self):
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._saved_signals[sig] = signal.signal(sig, self._interrupt)
            self._open()
            return self
        except BaseException:
            self.close()
            raise

    def _interrupt(self, _signum, _frame):
        if self._dispatch_critical or self._closing:
            self._cancel_requested = True
            return
        raise KeyboardInterrupt

    def _open(self):
        if hashlib.sha256(FIRMWARE.read_bytes()).hexdigest() != FIRMWARE_SHA256:
            raise RuntimeError('Frame firmware differs from the verified callback ABI')
        self.lock = open('/tmp/depth-model-cdsp.lock', 'a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.socket = socket.socket(socket.AF_UNIX)
        self.socket.settimeout(2)
        self.socket.connect(b'\0dsp_service')
        self.socket.sendall(struct.pack('<I', 4))
        payload, ancillary, flags, _ = self.socket.recvmsg(64, socket.CMSG_SPACE(16))
        for level, typ, raw in ancillary:
            if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                received = array.array('i')
                received.frombytes(raw[:len(raw) - len(raw) % received.itemsize])
                self.fds.extend(received)
        if payload != b'ADSP' or len(self.fds) != 2 or flags & socket.MSG_CTRUNC:
            raise RuntimeError('unexpected dsp_service handshake')
        if os.fstat(self.fds[1]).st_size != DMA_BYTES:
            raise RuntimeError('CDSP DMA-buffer size differs from the verified layout')
        self.mem = mmap.mmap(self.fds[1], DMA_BYTES, flags=mmap.MAP_SHARED,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=0x80000000)
        if struct.unpack_from('<QQ', self.mem, 0xA90000) != (0x1614df651d03f542, 9):
            raise RuntimeError('unknown CDSP protocol header')
        if self.get(WORK + 24):
            raise RuntimeError('CDSP operations interface is already in use')
        self.wait(lambda: (self.get(CTRL), self.get(CTRL + 4), self.get(CTRL + 8)) == (1, 0, 1), 5)
        # Identify the one eye client before interrupting it, retaining its exact
        # command, working directory and CPU affinity for restoration.
        proc = subprocess.run(['pgrep', '-x', 'eyetracking'], capture_output=True, text=True)
        pids = [int(p) for p in proc.stdout.split()
                if _process_state(p) not in (None,'Z')]
        if len(pids) > 1:
            raise RuntimeError('multiple eye clients are running')
        if pids:
            pid = pids[0]
            cmd = [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
            cmd[0] = os.readlink(f'/proc/{pid}/exe')
            cwd = os.readlink(f'/proc/{pid}/cwd')
            affinity = sorted(os.sched_getaffinity(pid))
            self.original_eye = cmd, cwd, affinity
            self.eye_pid = pid
            # Keeping the process alive prevents SteamVR's supervisor from
            # respawning a competing eye client during a long inference.
            self.eye_stopped = True
            os.kill(pid, signal.SIGSTOP)
            self.eye_stat_fd=os.open(f'/proc/{pid}/stat',os.O_RDONLY|os.O_CLOEXEC)
            self.wait(lambda: _process_state(pid) == 'T', 5)
            time.sleep(0.2)
        self.wait(lambda: (self.get(CTRL), self.get(CTRL + 4), self.get(CTRL + 8)) == (1, 0, 1), 5)
        if self.get(WORK + 24):
            raise RuntimeError('another DSP worker callback is installed')
        code_offset = CODE_BASE - 0x87000000
        self.saved_code = self.mem[code_offset:code_offset + CODE_BYTES]
        if any(self.saved_code) and self.saved_code != self.code:
            raise RuntimeError('dedicated callback code page is occupied')
        payload_offset = PAYLOAD_BASE - 0x87000000
        self.saved_payload = self.mem[payload_offset:payload_offset + PAYLOAD_BYTES]
        if any(self.saved_payload):
            raise RuntimeError('payload region is occupied; no payload was installed')
        if any(self.mem[ARENA_OFFSET:ARENA_OFFSET + ARENA_BYTES]):
            raise RuntimeError('dedicated matrix arena is occupied; no payload was installed')
        history_path = Path.home() / '.cache/depth-model/code-pages.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else {}
        key = f'{FIRMWARE_SHA256}:{CODE_BASE:#x}'
        if history.get(key, self.callback_sha256) != self.callback_sha256:
            raise RuntimeError('this code page has run a different payload; use a fresh page or explicit I-cache invalidation')
        history[key] = self.callback_sha256
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2) + '\n')
        self.saved_work = self.mem[WORK:WORK + 32]
        self.arena = memoryview(self.mem)[ARENA_OFFSET:ARENA_OFFSET + ARENA_BYTES]
        import ctypes
        self.arena_address = ctypes.addressof(ctypes.c_uint8.from_buffer(self.arena))
        self.mem[payload_offset:payload_offset + PAYLOAD_BYTES] = self.payload
        self.mem[code_offset:code_offset + CODE_BYTES] = self.code
        self.installed = True

    def matmul(self, a, b):
        spec = None
        def prepare(sequence):
            nonlocal spec
            spec = pack(self.arena, a, b, sequence)
        values = self.invoke(prepare, self.workers_done, lambda: result(self.arena, spec, self.sequence))
        self.bytes_uploaded += spec[-1]
        return values

    def conv(self, x, w):
        """Exact int32 sum(x*w) for uint8 x [H,W,C], int8 w [kh,kw,C,N]."""
        layout = None
        def prepare(sequence):
            nonlocal layout
            layout = pack_conv(self.arena, x, w, sequence, self.fence, self.raise_clock, self.packet_native, self.weight_cache)
        values = self.submit(prepare, lambda: conv_result(self.arena, layout, self.sequence, self.packet_native))
        self.bytes_uploaded += layout['xb'] + layout['wb']
        return values

    def conv_quantized(self,x,w,quantization):
        layout=None
        import numpy as np
        immutable=not w.flags.writeable and not quantization.flags.writeable
        key=(w.ctypes.data,w.shape,x.shape,quantization.ctypes.data)
        cached=self.hmx_layout_cache.get(key) if immutable else None
        if cached is not None:
            hmx=cached[2]
        else:
            hmx=self.use_hmx and w.shape[-1]%32==0 and bool(np.all(quantization[:,2]<=15))
            if hmx:
                shifts=quantization[:,2].astype(np.int64)
                half=np.where(shifts>0,np.int64(1)<<np.maximum(shifts-1,0),0)
                hmx=bool(np.all(quantization[:,0].astype(np.int64)-half>=-(2**31)))
            if hmx:
                try:conv_layout(x.shape,w.shape,True,True)
                except ValueError:hmx=False
            if immutable:self.hmx_layout_cache[key]=(w,quantization,hmx)
        def prepare(sequence):
            nonlocal layout
            layout=pack_conv(self.arena,x,w,sequence,self.fence,self.raise_clock,self.packet_native,self.weight_cache,quantization,hmx)
        values=self.submit(prepare,lambda:conv_result(self.arena,layout,self.sequence,self.packet_native))
        if hmx:
            ticks=struct.unpack_from('<12I',self.arena,2048)
            for phase in range(4):self.hmx_phase_cycles[phase]+=max(ticks[phase+4*t] for t in range(3))
        self.hmx_calls+=int(hmx)
        self.bytes_uploaded+=layout['xb']+layout['wb']+quantization.nbytes
        return values

    def workers_exited(self):
        # CTRL reaches 3 while workers may still be inside the payload, so only
        # these markers show that no worker is executing payload code.
        return all(STATUS.unpack_from(self.arena, STATUS_OFFSET + i * STATUS.size)[3] == EXITED
                   for i in range(3))

    def _persistent_done(self):
        if self.workers_done():
            return True
        if self.workers_exited():
            markers = [STATUS.unpack_from(self.arena, STATUS_OFFSET + i * STATUS.size)
                       for i in range(3)]
            command = struct.unpack_from('<9I', self.arena, 0)
            control = (self.get(CTRL), self.get(CTRL + 4), self.get(CTRL + 8))
            raise RuntimeError(f'DSP workers exited before command {self.sequence}: '
                               f'command={command} markers={markers} control={control}')
        return False

    def _release_workers(self):
        self.put(CTRL + 8, 1)
        self.put(CTRL + 4, 0)
        self.wait(lambda: self.get(CTRL) == 1)
        self.job_active = self.persistent = False

    def submit(self, prepare, collect):
        """Persistent payload command. The first wakes the workers with FLOPS;
        afterwards they poll the arena, so each command costs no wake-up."""
        if not self._eye_is_paused():
            raise RuntimeError('paused eye client changed state during inference')
        if self.persistent and self.workers_exited():
            # Workers exited on their idle timeout; release and wake them again.
            self.wait(lambda: self.get(CTRL) in (1, 3))
            self._release_workers()
        if self.persistent and not self.registered:
            raise RuntimeError('persistent workers exist without a registered callback')
        if not self.persistent and (self.job_active or self.get(CTRL) != 1):
            raise RuntimeError('previous DSP job is still active')
        if self.sequence >= 0xffffffff:
            raise RuntimeError('DSP command sequence exhausted')
        self.sequence += 1
        start = time.monotonic()
        prepare(self.sequence)
        prepared = time.monotonic()
        self.prepare_seconds += prepared - start
        if not self.persistent:
            self._dispatch_critical = True
            blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            try:
                self.fence()
                self._register_callback()
                self.put(CTRL + 4, 1)
                self.put(CTRL + 8, 4)
                self.job_active = self.persistent = True
                self.socket.sendall(b'work')
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
                self._dispatch_critical = False
            if self._cancel_requested:
                self._cancel_requested = False
                raise KeyboardInterrupt
        waiting = time.monotonic()
        if self.packet_native.qdm_poll(self.arena_address, self.sequence, 20000):
            self.wait(self._persistent_done, spin=0.05)
        completed = time.monotonic()
        self.wait_seconds += completed - waiting
        self.fence()
        values = collect()
        self.collect_seconds += time.monotonic() - completed
        self.calls += 1
        self.dispatch_seconds += time.monotonic() - start
        return values

    def stop_persistent(self):
        if not self.persistent:
            return
        if not self.workers_exited():
            self.sequence += 1
            pack_exit(self.arena, self.sequence, self.fence)
            self.wait(self.workers_exited)
        self.wait(lambda: self.get(CTRL) in (1, 3))
        self._release_workers()

    def invoke(self, prepare, done, collect):
        """Publish one command via `prepare(sequence)`, wake the workers, wait
        for `done()`, and return `collect()` before releasing the workers."""
        if self.job_active or self.get(CTRL) != 1:
            raise RuntimeError('previous DSP job is still active')
        if self.sequence >= 0xffffffff:
            raise RuntimeError('DSP command sequence exhausted')
        if not self._eye_is_paused():
            raise RuntimeError('paused eye client changed state during inference')
        self.sequence += 1
        prepare(self.sequence)
        start = time.monotonic()
        self._dispatch_critical = True
        blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            self.fence()
            self._register_callback()
            self.put(CTRL + 4, 1)
            self.put(CTRL + 8, 4)
            self.job_active = True
            self.socket.sendall(b'work')
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            self._dispatch_critical = False
        if self._cancel_requested:
            self._cancel_requested = False
            raise KeyboardInterrupt
        self.wait(lambda: self.get(CTRL) in (1, 3) and done())
        values = collect()
        self.put(CTRL + 8, 1)
        self.put(CTRL + 4, 0)
        self.wait(lambda: self.get(CTRL) == 1)
        self.job_active = False
        self.calls += 1
        self.dispatch_seconds += time.monotonic() - start
        return values

    def close(self):
        self._closing = True
        try:
            self._close()
        finally:
            self._closing = False
            if self.eye_stat_fd>=0:
                os.close(self.eye_stat_fd);self.eye_stat_fd=-1
            for sig, previous in self._saved_signals.items():
                signal.signal(sig, previous)
            self._saved_signals.clear()
            if self.lock is not None:
                self.lock.close(); self.lock = None

    def _close(self):
        # Do not send an EXIT or touch session memory if control is active but
        # this host has no owned dispatch in progress.
        if self.mem is not None and not self.job_active and not self._retain_reason:
            control = self._control()
            if control != (1, 0, 1):
                self.retain_session(f'unexpected CTRL state without a host job: {control}')
                self._stuck = True
        if self.mem is not None and self.persistent and not self._retain_reason and self.registered and self.get(WORK + 24) == CODE_BASE:
            try:
                self.stop_persistent()
            except TimeoutError:
                # job_active stays set, so nothing below restores code or gaze.
                print('Persistent DSP workers did not exit. Callback/arena retained; eye client was not restarted.', flush=True)
                self._stuck = True
        if self.arena is not None and self.registered and self.get(WORK+24)==CODE_BASE:
            axim=struct.unpack_from('<I',self.arena,256)[0]
            hmx=struct.unpack_from('<I',self.arena,512)[0]
            if axim==0x41584931 or hmx==0x484d5831:
                self._stuck=True;self.job_active=True
                print('DSP clock/power restore incomplete; callback and saved register state retained.',flush=True)
        # Recheck after stopping our own idle persistent workers.
        if self.mem is not None and not self.job_active and not self._retain_reason:
            control = self._control()
            if control != (1, 0, 1):
                self.retain_session(f'unexpected CTRL state without a host job: {control}')
                self._stuck = True

        pointer = self.get(WORK + 24) if self.mem is not None else 0
        if self.mem is not None and self.registered and pointer not in (0, CODE_BASE) and not self._retain_reason:
            self.retain_session(f'unexpected callback pointer during cleanup: {pointer:#x}')
            self._stuck = True
        if self.mem is not None and self.installed and not self.registered and pointer != 0 and not self._retain_reason:
            self.retain_session(f'callback pointer changed before this host registered it: {pointer:#x}')
            self._stuck = True

        safe = not self.job_active and not self._retain_reason and not self._stuck
        if self.mem is not None and self.registered and not self._retain_reason and pointer == 0:
            off = CODE_BASE - 0x87000000
            fresh = (struct.unpack_from('<QQ', self.mem, 0xA90000) == (0x1614df651d03f542, 9)
                     and self.get(CTRL) == 1
                     and (not any(self.mem[off:off + CODE_BYTES]) or self.mem[off:off + CODE_BYTES] == self.code))
            if fresh:
                # The stock suspend/resume path rebooted the DSP. These regions
                # belong to the new session: release our mappings without writes.
                self.job_active = self.registered = self.installed = False
                safe = not self._stuck
                print('Detected a fresh DSP after suspend/reset; original memory was not restored over it.', flush=True)
            elif not self._retain_reason:
                self.retain_session('registered callback disappeared without a verified DSP reset')
                self._stuck = True
                safe = False
        if self.mem is not None and self.job_active and self.arena is not None and not self._stuck:
            # Never clear memory while a worker might still be executing it.
            try:
                self.wait(lambda: self.get(CTRL) in (1, 3) and self.workers_done())
                self.put(CTRL + 8, 1); self.put(CTRL + 4, 0)
                self.wait(lambda: self.get(CTRL) == 1)
                self.job_active = False
                safe = True
            except TimeoutError:
                print('DSP workers have not quiesced. Callback/arena retained; eye client was not restarted.', flush=True)
        if self.mem is not None and self.installed and safe and self._control() == (1, 0, 1):
            if self.registered:
                self.put(WORK + 24, 0)
            self.mem[WORK:WORK + 32] = self.saved_work
            off = CODE_BASE - 0x87000000
            self.mem[off:off + CODE_BYTES] = self.saved_code
            poff = PAYLOAD_BASE - 0x87000000
            self.mem[poff:poff + PAYLOAD_BYTES] = self.saved_payload
            self.arena[:] = bytes(ARENA_BYTES)
            self.registered = self.installed = False
        if self.arena is not None:
            self.arena.release(); self.arena = None
        if self.mem is not None:
            self.mem.close(); self.mem = None
        if self.socket is not None:
            self.socket.close(); self.socket = None
        for fd in self.fds:
            os.close(fd)
        self.fds = []
        if self._retain_reason:
            print(f'DSP session retained for recovery: {self._retain_reason}. Callback data and paused eye client were preserved.', flush=True)
        if safe and self.eye_stopped and self.original_eye:
            # FLOPS reuses gaze buffers, so resume via a fresh client that
            # uploads its weights again, rather than SIGCONT alone.
            eye_path = Path(f'/proc/{self.eye_pid}') if self.eye_pid is not None else None
            if eye_path is not None and _process_state(self.eye_pid) not in (None,'Z'):
                try:
                    os.kill(self.eye_pid, signal.SIGTERM)
                    os.kill(self.eye_pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                self.wait(lambda: _process_state(self.eye_pid) in (None,'Z'), 5)
            # Do not launch a duplicate if another supervisor already restored it.
            # On this Frame the supervisor does not replace eyetracking within
            # three seconds. A short grace period still avoids racing a prompt
            # replacement, without charging every NPU burst three idle seconds.
            deadline = time.monotonic() + 0.35
            while True:
                existing = subprocess.run(['pgrep', '-x', 'eyetracking'], capture_output=True, text=True)
                live = [int(p) for p in existing.stdout.split()
                        if _process_state(p) not in (None,'Z')]
                if live or time.monotonic() >= deadline:
                    break
                time.sleep(0.1)
            if not live:
                cmd, cwd, affinity = self.original_eye
                log_path = Path.home() / 'frame-dsp/depth-model-eye-restored.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open('ab') as log:
                    # Double fork so a long-lived chat process never owns a
                    # zombie eye client after SteamVR restarts it.
                    read_fd, write_fd = os.pipe()
                    child_pid = os.fork()
                    if child_pid == 0:
                        os.close(read_fd)
                        grandchild_pid = os.fork()
                        if grandchild_pid:
                            os.write(write_fd, str(grandchild_pid).encode())
                            os._exit(0)
                        os.close(write_fd)
                        os.setsid()
                        os.chdir(cwd)
                        os.sched_setaffinity(0, set(affinity))
                        os.dup2(log.fileno(), 1)
                        os.dup2(log.fileno(), 2)
                        os.execv(cmd[0], cmd)
                        os._exit(127)
                    os.close(write_fd)
                    restored_pid = int(os.read(read_fd, 32))
                    os.close(read_fd)
                    os.waitpid(child_pid, 0)
                time.sleep(1)
                restored = Path(f'/proc/{restored_pid}')
                if _process_state(restored_pid) in (None,'Z'):
                    # SteamVR may replace a client while it starts. Accept its
                    # live replacement, but never leave tracking absent.
                    replacement = _wait_for_single_eye()
                    if replacement is None:
                        raise RuntimeError(f'eye client did not stay running; see {log_path}')
                    restored_pid = replacement
                print(f'Restored eye client {restored_pid}', flush=True)
            self.eye_stopped = False
        if self.lock is not None:
            self.lock.close(); self.lock = None
        if self._retain_reason:
            raise RuntimeError(f'DSP state retained; explicit recovery is required: {self._retain_reason}')
        if self._stuck:
            raise RuntimeError('CDSP worker or clock cleanup incomplete; callback retained and eye tracking paused. Reboot the Frame before another NPU session.')

    def __exit__(self, *args):
        self.close()
