# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise the real probe callbacks with a fake transport; no device access."""
import contextlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import run_adapter_frame_probe as probe


class ProbeHostTests(unittest.TestCase):
    def exercise(self, fault=None):
        fixture = (probe.HERE / 'fixture-qdrx-1195.bin').read_bytes()
        simulated_output = (probe.HERE / 'output-qdrx-1195.bin').read_bytes()[36:]
        instances = []

        class Transport:
            def __init__(self, *_):
                self.arena = bytearray(probe.ARENA_BYTES)
                self.payload_sha256 = 'mock'
                self.retained = False
                instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def fence(self):
                pass

            def retain_session(self, reason):
                self.retained = bool(reason)

            def invoke(self, prepare, wait, collect):
                prepare(7)
                packet = struct.unpack_from('<10I', self.arena)
                assert packet == (probe.MAGIC, 1, 7, 8, 64, 128, 128, 1, 0, 0x700000)
                words = [0] * 32
                words[0], words[17], words[18] = probe.MAGIC, 1, 1
                words[19:23] = probe.EXPECTED_TLB
                words[23:27] = probe.EXPECTED_TLB
                if fault == 'restore':
                    words[17] = 0
                elif fault == 'mapping':
                    words[23] ^= 1
                elif fault == 'clock_before_backup':
                    words[11], words[17], words[18] = 1, 0, 0
                struct.pack_into('<32I', self.arena, 2048, *words)
                self.arena[probe.OUTPUT_OFFSET:probe.OUTPUT_OFFSET + len(simulated_output)] = simulated_output
                self.arena[probe.ECHO_OFFSET:probe.ECHO_OFFSET + 65536] = fixture[36:36 + 65536]
                return collect()

        with tempfile.TemporaryDirectory() as temp:
            argv = ['probe', '--report', str(Path(temp) / 'report.json')]
            with patch.object(probe, 'FrameTransport', Transport), \
                 patch.object(probe, 'snapshot', lambda: {'mock': True}), \
                 patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                if fault:
                    with self.assertRaisesRegex(RuntimeError, 'restore/power verification failed'):
                        probe.main()
                else:
                    probe.main()
            self.assertEqual(instances[0].retained, fault is not None)

    def test_valid_restore(self):
        self.exercise()

    def test_failed_restore_is_retained(self):
        self.exercise('restore')

    def test_changed_mapping_is_retained(self):
        self.exercise('mapping')

    def test_clock_failure_before_backup_is_retained(self):
        self.exercise('clock_before_backup')


if __name__ == '__main__':
    unittest.main()
