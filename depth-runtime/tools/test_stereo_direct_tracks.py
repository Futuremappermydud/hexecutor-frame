"""Known image motion controls for direct stereo track propagation."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from stereo_direct_tracks import track_window


class DirectTrackTests(unittest.TestCase):
    def fixture(self, occluded=False):
        rng = np.random.default_rng(831)
        texture = cv2.GaussianBlur(rng.integers(30, 220, (256, 256), dtype=np.uint8), (3, 3), .65)
        points = np.array([[x, y] for y in (70., 110., 150., 190.) for x in (80., 120., 160., 200.)])
        frames = []
        motion = np.array([3.25, -1.5])
        for frame in range(4):
            images = []
            for eye in range(2):
                delta = frame*motion + [-12.*eye, 0.]
                image = cv2.warpAffine(texture, np.float32([[1, 0, delta[0]], [0, 1, delta[1]]]), (256, 256))
                if eye:
                    image = np.rint(image.astype(float)*.7 + 20).astype(np.uint8)
                if occluded and frame >= 2:
                    image[:] = 100
                images.append(image)
            # Deliberately no ORB/stereo redetections after the first frame.
            stereo = np.column_stack((points, points + [-12., 0.])) if frame == 0 else np.empty((0, 4))
            frames.append({'images': images, 'masks': [np.full((256, 256), 255, np.uint8) for _ in range(2)], 'stereo': stereo})
        return frames, motion

    def test_no_redetection_and_known_two_eye_motion(self):
        frames, motion = self.fixture()
        rows, diagnostics = track_window(frames)
        complete = [rows[rows[:, 1] == t] for t in np.unique(rows[:, 1]) if np.sum(rows[:, 1] == t) == 4]
        self.assertGreaterEqual(len(complete), 12)
        for track in complete:
            delta = track[:, 0, None] * motion
            np.testing.assert_allclose(track[:, 2:4] - track[0, 2:4], delta, atol=.2)
            np.testing.assert_allclose(track[:, 4:6] - track[0, 4:6], delta, atol=.2)
        self.assertEqual(diagnostics['birth_counts'][1:], [0, 0, 0])

    def test_birth_anchor_removes_injected_temporal_coordinate_drift(self):
        frames, motion = self.fixture()
        def drifting(a, b, ma, mb, points):
            return np.asarray(points) + motion + [.15, 0.], np.arange(len(points))
        with patch('stereo_direct_tracks.propagate', side_effect=drifting):
            chained, _ = track_window(frames)
            anchored, diagnostic = track_window(frames, anchor_refinement=True)
        self.assertTrue(diagnostic['anchor_refinement'])
        self.assertGreaterEqual(np.sum(anchored[:,0] == 3), 12)
        self.assertGreater(np.mean(chained[chained[:,0] == 3,2]-chained[chained[:,0] == 0,2])-3*motion[0], .4)
        for t in np.unique(anchored[:,1]):
            track=anchored[anchored[:,1] == t]
            np.testing.assert_allclose(track[:,2:4]-track[0,2:4],track[:,0,None]*motion,atol=.15)
            np.testing.assert_allclose(track[:,4:6]-track[0,4:6],track[:,0,None]*motion,atol=.15)

    def test_occlusion_ends_tracks(self):
        frames, _ = self.fixture(occluded=True)
        rows, _ = track_window(frames)
        self.assertGreater(len(rows), 0)
        self.assertTrue(np.all(rows[:, 0] < 2))

    def test_masked_destination_ends_tracks(self):
        frames, _ = self.fixture()
        frames[1]['masks'][1][:] = 0
        rows, _ = track_window(frames)
        self.assertEqual(len(rows), 0)

    def test_empty_window(self):
        rows, _ = track_window([])
        self.assertEqual(rows.shape, (0, 6))


if __name__ == '__main__':
    unittest.main()
