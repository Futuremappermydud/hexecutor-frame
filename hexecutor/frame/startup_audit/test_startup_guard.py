"""Host-only truth table for the proposed QDS callback-start guard.

Run with: python3 frame/startup_audit/test_startup_guard.py
This models values only; it does not emulate Hexagon memory ordering or DSPs.
"""
import unittest

QDS_MAGIC = 0x32564451
QDS_VERSION = 6
QDS_GRAPH_RUN = 3


def valid_start(words, sequence_after, config_valid=True):
    """Model seven header words plus a second sequence read after a barrier."""
    magic, version, sequence_before, op, first, last, flags = words
    return bool(
        config_valid
        and sequence_before != 0
        and sequence_before == sequence_after
        and magic == QDS_MAGIC
        and op == QDS_GRAPH_RUN
        and version == QDS_VERSION
        and first <= last
        and flags & ~1 == 0
    )


class StartupGuardTests(unittest.TestCase):
    def setUp(self):
        self.header = [QDS_MAGIC, QDS_VERSION, 1, QDS_GRAPH_RUN, 0, 5068, 0]

    def test_valid_committed_graph_run(self):
        self.assertTrue(valid_start(self.header, 1))

    def test_all_zero_startup_is_rejected(self):
        self.assertFalse(valid_start([0] * 7, 0))

    def test_zero_sequence_is_rejected_even_with_other_valid_fields(self):
        h = self.header.copy()
        h[2] = 0
        self.assertFalse(valid_start(h, 0))

    def test_sequence_must_be_stable(self):
        self.assertFalse(valid_start(self.header, 2))

    def test_header_magic_op_and_version_are_checked(self):
        for index, bad in ((0, 0), (1, 8), (3, 2)):
            with self.subTest(index=index):
                h = self.header.copy()
                h[index] = bad
                self.assertFalse(valid_start(h, 1))

    def test_range_must_be_ordered(self):
        h = self.header.copy()
        h[4:6] = [3, 2]
        self.assertFalse(valid_start(h, 1))

    def test_only_direct_flag_is_permitted(self):
        h = self.header.copy()
        h[6] = 2
        self.assertFalse(valid_start(h, 1))

    def test_config_must_be_valid_before_mapping(self):
        self.assertFalse(valid_start(self.header, 1, config_valid=False))


if __name__ == "__main__":
    unittest.main()
