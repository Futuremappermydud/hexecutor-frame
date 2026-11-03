#!/usr/bin/env python3
"""Native analyzer regression: guarded ages, baseline compatibility, invalid policy."""
import csv
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ANALYZER = Path('bin/analyze_occlusion_benchmark').resolve()


def recording(path, flags):
    n = 64
    identity = (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    data = b'FROCCB02' + struct.pack('<II18f', 8, 8, 10., 4., 4., *([0.] * 12), 0., 8., 3.)
    for i, flag in enumerate(flags):
        ns = 1_000_000_000 + i * 800_000_000
        meta = struct.pack('<QQQQqIIf', i+1, ns, ns+100, ns+200, 0, flag, 1, .8)
        meta += struct.pack('<30f', *([0.] * 11), 0., 0., 0., 0., 0., 0., 1., *identity, 0., 0., 0.)
        assert len(meta) == 172
        data += struct.pack('<II', 0x324d5246, 172 + n*11) + meta
        data += bytes([128] * (2*n)) + struct.pack('<' + 'f' * (2*n), *([1.] * (2*n))) + bytes([2]*n)
    data += struct.pack('<IIII', 0x32444e45, len(flags), 0, 1)
    path.write_bytes(data)


class Replay(unittest.TestCase):
    def run_case(self, flags):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording(root / 'recording.bin', flags)
            result = subprocess.run([str(ANALYZER), str(root/'recording.bin'), str(root/'frames.csv')],
                                    capture_output=True, text=True)
            rows = []
            if (root/'frames.csv').exists():
                with (root/'frames.csv').open() as source:
                    rows = list(csv.DictReader(source))
            return result, rows

    def test_retention(self):
        baseline, b = self.run_case([3]*3)
        guarded, g = self.run_case([19]*3)
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(guarded.returncode, 0, guarded.stderr)
        self.assertIn('legacy_max_age_ms=2500', guarded.stdout)
        self.assertEqual(float(b[-1]['legacy_mesh_pct']), 0)
        self.assertGreater(float(g[-1]['legacy_mesh_pct']), 0)

    def test_pose_loss(self):
        result, rows = self.run_case([19, 19, 16])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(float(rows[-1]['legacy_mesh_pct']), 0)

    def test_mixed_policy_rejected(self):
        for flags in ([19, 3], [27]):
            result, _ = self.run_case(flags)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('retention policy', result.stderr)


if __name__ == '__main__':
    unittest.main()
