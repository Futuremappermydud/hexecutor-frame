#!/usr/bin/env python3
"""Synthetic two-dimensional subpixel flow and mask safety checks."""
import copy
import unittest

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

if cv2 is not None:
    from evaluate_stereo_self_calibration import refine_matches, tag_prediction
    from stereo_self_calibration import fit_calibration
    from test_stereo_self_calibration import general_scene, K, ROTATION, FOCAL_SCALE, rotation_matrix


@unittest.skipIf(cv2 is None, 'OpenCV is required for offline matcher checks')
class MatchRefinementTests(unittest.TestCase):
    def test_subpixel_xy_and_mask_exclusion(self):
        # Frozen tolerance 0.15 px per axis, set before running refinement.
        rng = np.random.default_rng(115)
        left = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), 0.65)
        displacement = np.array([1.25, -0.625])
        right = cv2.warpAffine(left, np.float32([[1., 0., displacement[0]],
                                                [0., 1., displacement[1]]]), (256, 256))
        locations = np.array([[x, y] for y in range(65, 201, 30)
                              for x in range(65, 201, 30)], dtype=float)
        seeds = np.column_stack((locations, locations + [1., -1.]))
        masks = [np.full(left.shape, 255, dtype=np.uint8) for _ in range(2)]
        refined = refine_matches([left, right], masks, seeds)
        self.assertGreaterEqual(len(refined), 20)
        errors = refined[:, 2:] - refined[:, :2] - displacement
        self.assertLess(np.max(np.abs(errors)), 0.15)
        # A masked right seed must be excluded even when optical flow converges.
        masked = [m.copy() for m in masks]
        x, y = np.rint(seeds[0, 2:]).astype(int)
        masked[1][y, x] = 0
        refined_masked = refine_matches([left, right], masked, seeds[:1])
        self.assertEqual(len(refined_masked), 0)

    def test_tag_scoring_is_invariant_to_rectification_gauge(self):
        _, baseline, matches = general_scene()
        fitted = fit_calibration(matches, K, K, model='full6')
        self.assertTrue(fitted['accepted'], fitted)
        size = 0.12
        obj = np.array([[-size/2, -size/2, 0.], [size/2, -size/2, 0.],
                        [size/2, size/2, 0.], [-size/2, size/2, 0.]])
        points = obj @ rotation_matrix([0.25, -0.2, 0.05]).T + [0.08, -0.06, 1.4]
        right = (points - baseline) @ rotation_matrix(ROTATION)
        def px(p, scale=1.):
            return np.column_stack((K[0, 0] * scale * p[:, 0] / p[:, 2] + K[0, 2],
                                    K[1, 1] * scale * p[:, 1] / p[:, 2] + K[1, 2]))
        observations = [{'left': px(points), 'right': px(right, FOCAL_SCALE)}]
        score = tag_prediction(observations, fitted, K, np.linalg.norm(baseline), size)
        self.assertLess(score['right_error_x_px']['p90_abs'], 0.02)
        self.assertLess(score['right_error_y_px']['p90_abs'], 0.02)
        changed = copy.deepcopy(fitted)
        changed['parameters']['rectification_rotation_vector_rad'] = [0.08, -0.03, 0.01]
        self.assertEqual(score, tag_prediction(observations, changed, K, np.linalg.norm(baseline), size))

    def test_far_seed_and_empty_input(self):
        rng = np.random.default_rng(118)
        image = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        mask = np.full(image.shape, 255, dtype=np.uint8)
        empty = refine_matches([image, image], [mask, mask], np.empty((0, 4)))
        self.assertEqual(empty.shape, (0, 4))
        # Large initial mismatch is not allowed to become an accepted update.
        result = refine_matches([image, image], [mask, mask], [[64., 64., 70., 70.]])
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
