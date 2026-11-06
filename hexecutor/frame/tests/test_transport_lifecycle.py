"""Host mocks for the shared FrameTransport lifecycle; no device access."""
import sys
import struct
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frame"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Keep this lifecycle test independent of NumPy and target protocol binaries.
conv_stub = types.ModuleType("conv_protocol")
conv_stub.EXITED = 0x54495845
conv_stub.conv_layout = conv_stub.conv_result = conv_stub.pack_conv = conv_stub.pack_exit = None
conv_stub.fence_library = lambda *_args: (lambda: None)
hvx_stub = types.ModuleType("hvx_protocol")
hvx_stub.ARENA_BYTES = 0x800000
hvx_stub.ARENA_OFFSET = 0x1800000
hvx_stub.CODE_BASE = 0x87606000
hvx_stub.CODE_BYTES = 4096
hvx_stub.PAYLOAD_BASE = 0x87610000
hvx_stub.PAYLOAD_BYTES = 0x8000
hvx_stub.STATUS = struct.Struct('<4I')
hvx_stub.STATUS_OFFSET = 128
hvx_stub.pack = hvx_stub.result = None
import importlib.util
spec=importlib.util.spec_from_file_location("transport_lifecycle_under_test",ROOT/"frame/frame_transport.py")
candidate=importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules,{"conv_protocol":conv_stub,"hvx_protocol":hvx_stub}):
    spec.loader.exec_module(candidate)


class Memory(bytearray):
    def close(self):
        self.closed = True


class FakeSocket:
    def __init__(self, transport, events):
        self.transport = transport
        self.events = events

    def sendall(self, value):
        self.events.append(("socket_send", self.transport.get(candidate.WORK + 24)))
        self.transport.put(candidate.CTRL, 1)
        self.transport.put(candidate.CTRL + 4, 0)
        self.transport.put(candidate.CTRL + 8, 1)
        for i in range(3):
            candidate.STATUS.pack_into(self.transport.arena,
                                       candidate.STATUS_OFFSET + i * candidate.STATUS.size,
                                       self.transport.sequence, 0, 0, 0)

    def close(self):
        pass


class FakeNative:
    @staticmethod
    def qdm_poll(*_args):
        return 0


def make_transport():
    t = object.__new__(candidate.FrameTransport)
    t.mem = Memory(candidate.DMA_BYTES)
    t.arena = memoryview(t.mem)[candidate.ARENA_OFFSET:
                                candidate.ARENA_OFFSET + candidate.ARENA_BYTES]
    t.code = b'C' * candidate.CODE_BYTES
    code_off = candidate.CODE_BASE - 0x87000000
    t.mem[code_off:code_off + candidate.CODE_BYTES] = t.code
    t.payload = b'P' * candidate.PAYLOAD_BYTES
    payload_off = candidate.PAYLOAD_BASE - 0x87000000
    t.mem[payload_off:payload_off + candidate.PAYLOAD_BYTES] = t.payload
    t.saved_code = b'S' * candidate.CODE_BYTES
    t.saved_payload = b'Q' * candidate.PAYLOAD_BYTES
    t.saved_work = bytes(32)
    t.put(candidate.WORK + 24, 0)
    t.put(candidate.CTRL, 1)
    t.put(candidate.CTRL + 4, 0)
    t.put(candidate.CTRL + 8, 1)
    t.installed = True
    t.registered = False
    t.job_active = False
    t.persistent = False
    t.sequence = 0
    t.eye_pid = None
    t.eye_stat_fd = -1
    t.fds = []
    t.socket = None
    t.lock = None
    t.eye_stopped = False
    t.original_eye = None
    t._saved_signals = {}
    t._dispatch_critical = t._closing = t._cancel_requested = False
    t._stuck = False
    t._retain_reason = None
    t.timeout = 1
    t.fence_events = []
    t.fence = lambda: t.fence_events.append("fence")
    t.packet_native = FakeNative()
    t.arena_address = 1
    t.prepare_seconds = t.wait_seconds = t.collect_seconds = t.dispatch_seconds = 0.0
    t.calls = t.bytes_uploaded = 0
    t.events = []
    t.socket = FakeSocket(t, t.events)
    return t


class FrameTransportCandidateTests(unittest.TestCase):
    def test_loaded_context_is_not_registered_before_first_prepare(self):
        t = make_transport()
        self.assertTrue(t.installed)
        self.assertFalse(t.registered)
        self.assertEqual(t.get(candidate.WORK + 24), 0)

    def test_submit_registers_after_prepare_and_fence(self):
        t = make_transport()
        def prepare(seq):
            t.events.append(("prepare", seq, t.get(candidate.WORK + 24)))
        with mock.patch.object(candidate.signal, "pthread_sigmask", return_value=set()):
            t.submit(prepare, lambda: "collected")
        names = [event[0] for event in t.events]
        self.assertEqual(names, ["prepare", "socket_send"])
        self.assertEqual(t.events[0][2], 0)
        self.assertEqual(t.events[1][1], candidate.CODE_BASE)
        self.assertTrue(t.registered)
        self.assertIn("fence", t.fence_events)

    def test_invoke_registers_after_prepare_and_fence(self):
        t = make_transport()
        def prepare(seq):
            t.events.append(("prepare", seq, t.get(candidate.WORK + 24)))
        with mock.patch.object(candidate.signal, "pthread_sigmask", return_value=set()):
            value = t.invoke(prepare, lambda: True, lambda: "collected")
        self.assertEqual(value, "collected")
        self.assertEqual(t.events[0], ("prepare", 1, 0))
        self.assertEqual(t.events[1], ("socket_send", candidate.CODE_BASE))
        self.assertTrue(t.registered)
        self.assertFalse(t.job_active)

    def test_prepare_failure_never_registers_callback(self):
        t = make_transport()
        def bad_prepare(_seq):
            raise ValueError("mock prepare failure")
        with self.assertRaises(ValueError):
            t.submit(bad_prepare, lambda: None)
        self.assertFalse(t.registered)
        self.assertEqual(t.get(candidate.WORK + 24), 0)

    def test_registration_rejects_nonidle_control_without_writing_pointer(self):
        t = make_transport()
        t.put(candidate.CTRL, 2)
        t.put(candidate.CTRL + 8, 8)
        with self.assertRaisesRegex(RuntimeError, "verified idle"):
            t._register_callback()
        self.assertFalse(t.registered)
        self.assertEqual(t.get(candidate.WORK + 24), 0)

    def test_registration_rejects_eye_that_is_not_stopped(self):
        t = make_transport()
        t.eye_pid = 123
        t.eye_stat_fd = 44
        with mock.patch.object(candidate.os, "pread", return_value=b'123 (eyetracking) S'):
            with self.assertRaisesRegex(RuntimeError, "paused eye client changed state"):
                t._register_callback()
        self.assertFalse(t.registered)
        self.assertEqual(t.get(candidate.WORK + 24), 0)

    def test_unregistered_loaded_pages_restore_when_idle(self):
        t = make_transport()
        mem = t.mem
        code_off = candidate.CODE_BASE - 0x87000000
        payload_off = candidate.PAYLOAD_BASE - 0x87000000
        saved_work = bytearray(b'W' * 32)
        struct.pack_into('<I', saved_work, 24, 0)
        t.saved_work = bytes(saved_work)
        t.mem[code_off:code_off + candidate.CODE_BYTES] = t.code
        t.mem[payload_off:payload_off + candidate.PAYLOAD_BYTES] = t.payload
        t._close()
        self.assertFalse(t.installed)
        self.assertFalse(t.registered)
        self.assertEqual(struct.unpack_from('<I', mem, candidate.WORK + 24)[0], 0)
        self.assertEqual(mem[code_off:code_off + candidate.CODE_BYTES], t.saved_code)
        self.assertEqual(mem[payload_off:payload_off + candidate.PAYLOAD_BYTES], t.saved_payload)
        self.assertEqual(mem[candidate.WORK:candidate.WORK + 32], t.saved_work)

    def test_registered_callback_fresh_dsp_reset_does_not_restore_old_pages(self):
        t = make_transport()
        mem = t.mem
        t.registered = True
        t.put(candidate.WORK + 24, 0)
        t.mem[0xA90000:0xA90010] = bytes.fromhex('42f5031d65df14160900000000000000')
        code_off = candidate.CODE_BASE - 0x87000000
        t.mem[code_off:code_off + candidate.CODE_BYTES] = bytes(candidate.CODE_BYTES)
        t.arena[0] = 0x5A
        t._close()
        self.assertFalse(t.installed)
        self.assertFalse(t.registered)
        self.assertEqual(mem[code_off:code_off + candidate.CODE_BYTES], bytes(candidate.CODE_BYTES))
        self.assertEqual(mem[candidate.ARENA_OFFSET], 0x5A)

    def test_registered_callback_idle_cleanup_clears_pointer_and_restores_pages(self):
        t = make_transport()
        mem = t.mem
        t.registered = True
        t.put(candidate.WORK + 24, candidate.CODE_BASE)
        saved_work = bytearray(b'R' * 32)
        struct.pack_into('<I', saved_work, 24, 0)
        t.saved_work = bytes(saved_work)
        t._close()
        code_off = candidate.CODE_BASE - 0x87000000
        payload_off = candidate.PAYLOAD_BASE - 0x87000000
        self.assertEqual(struct.unpack_from('<I', mem, candidate.WORK + 24)[0], 0)
        self.assertEqual(mem[candidate.WORK:candidate.WORK + 32], t.saved_work)
        self.assertEqual(mem[code_off:code_off + candidate.CODE_BYTES], t.saved_code)
        self.assertEqual(mem[payload_off:payload_off + candidate.PAYLOAD_BYTES], t.saved_payload)

    def test_unexpected_active_control_with_no_host_job_retains_and_never_resumes_eye(self):
        t = make_transport()
        mem = t.mem
        t.eye_stopped = True
        t.original_eye = (["eye"], "/", [0])
        t.put(candidate.CTRL, 2)
        t.put(candidate.CTRL + 8, 8)
        with mock.patch.object(candidate.subprocess, "run", side_effect=AssertionError("eye restart attempted")):
            with self.assertRaisesRegex(RuntimeError, "state retained"):
                t._close()
        self.assertFalse(t.job_active)
        self.assertTrue(t._stuck)
        self.assertIn("unexpected CTRL", t._retain_reason)
        self.assertEqual(struct.unpack_from('<I', mem, candidate.WORK + 24)[0], 0)

    def test_unexpected_active_control_does_not_send_exit_to_persistent_workers(self):
        t = make_transport()
        t.registered = True
        t.persistent = True
        t.put(candidate.WORK + 24, candidate.CODE_BASE)
        t.put(candidate.CTRL, 2)
        t.put(candidate.CTRL + 8, 8)
        with mock.patch.object(t, "stop_persistent") as stop:
            with self.assertRaisesRegex(RuntimeError, "state retained"):
                t._close()
        stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
