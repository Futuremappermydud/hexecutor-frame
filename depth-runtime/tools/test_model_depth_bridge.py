"""Exercise native backend selection and the wire protocol without DSP hardware."""

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

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

import numpy as np

from depth_model import offline_bridge as bridge


class Socket:
    def __init__(self, temporal=False):
        self.input = bytearray(struct.pack('<4I', 0x36504451, 42, 256, 320))
        self.input.extend(bytes(64 + 2 * 256 * 320))
        if temporal:
            self.input.extend(bytes((2 * 320 * 256 + 32) * 4))
        self.output = []

    def recv_into(self, view):
        n = min(len(view), len(self.input))
        view[:n] = self.input[:n]
        del self.input[:n]
        return n

    def sendall(self, data):
        self.output.append(data)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass


class NativeBridgeTests(unittest.TestCase):
    def run_backend(self, args, expected_engine, model='primary', temporal=False):
        sock = Socket(temporal)
        disparity = np.stack([np.full((320, 256), n, np.float32) for n in (3, 7)])
        uncertainty = np.stack([np.full((320, 256), n, np.uint8) for n in (2, 4)])
        infer = mock.Mock(return_value=(disparity, uncertainty))
        sessions = []

        @contextlib.contextmanager
        def session(*values):
            sessions.append(values)
            yield infer

        argv = ['bridge', '--fd', '9', '--package', '/tmp/native-package', *args]
        old_path = sys.path[:]
        try:
            with mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(bridge, 'prepared_session', session), \
                 mock.patch.object(bridge.socket, 'socket', return_value=sock), \
                 mock.patch('select.select', return_value=([sock], [], [])), \
                 mock.patch.object(os, 'sched_setaffinity', create=True), \
                 contextlib.redirect_stdout(io.StringIO()):
                bridge.main()
        finally:
            sys.path[:] = old_path
        self.assertEqual(sessions, [(Path('/tmp/native-package').resolve(), model, *expected_engine)])
        self.assertEqual(struct.unpack('<4I', sock.output[0]), (0x36524451, 256, 320, 4 if temporal else 3))
        self.assertEqual(struct.unpack('<3If', sock.output[1])[:3], (0x364f4451, 42, 256 * 320))
        np.testing.assert_array_equal(np.frombuffer(sock.output[2], '<f4'), disparity[0].ravel())
        self.assertEqual(sock.output[3], uncertainty[0].tobytes())
        self.assertEqual(len(sock.output), 5 if temporal else 4)
        if temporal:
            self.assertEqual(sock.output[4], disparity[1].tobytes())
        infer.assert_called_once()
        images, hidden, transforms = infer.call_args.args
        self.assertEqual(images.shape, (2, 320, 256))
        self.assertEqual(hidden.shape, (1, 2, 320, 256))
        self.assertEqual(transforms.shape, (1, 2, 4, 4))

    def test_default_uses_qnn_and_preserves_protocol(self):
        self.run_backend([], (bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP))

    def test_explicit_qnn_with_temporal_and_second_plane(self):
        self.run_backend(['--backend', 'prepared-qnn', '--host-temporal', '--second-disparity'],
                         (bridge.QNN_ENGINE, bridge.QNN_BOOTSTRAP), temporal=True)

    def test_prepared_supports_alternate(self):
        self.run_backend(['--backend', 'prepared', '--model', 'alternate'], (), model='alternate')

    def assert_rejected(self, args):
        with mock.patch.object(sys, 'argv', ['bridge', '--fd', '9', '--package', '/tmp/package', *args]), \
             mock.patch.object(bridge.socket, 'socket') as socket_factory, \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as error:
            bridge.main()
        self.assertEqual(error.exception.code, 2)
        socket_factory.assert_not_called()

    def test_removed_hybrid_is_rejected(self):
        self.assert_rejected(['--backend', 'hybrid'])

    def test_qnn_rejects_alternate_before_opening_hardware(self):
        self.assert_rejected(['--model', 'alternate'])


if __name__ == '__main__':
    unittest.main()
