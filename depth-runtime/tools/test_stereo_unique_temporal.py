import unittest
from unittest.mock import patch
import stereo_direct_tracks as tracks
import evaluate_stereo_temporal_lens as temporal
from evaluate_stereo_unique_temporal import run


class UniqueReplayContextTests(unittest.TestCase):
    def test_original_matcher_and_track_builder_restore_after_failure(self):
        old_matcher, old_builder = tracks.refine_matches, temporal.track_window
        with patch.object(temporal, 'run', side_effect=RuntimeError('controlled replay failure')):
            with self.assertRaises(RuntimeError):
                run('unused', 'unused', 'unused', 'unused', 11.9)
        self.assertIs(tracks.refine_matches, old_matcher)
        self.assertIs(temporal.track_window, old_builder)


if __name__ == '__main__':
    unittest.main()
