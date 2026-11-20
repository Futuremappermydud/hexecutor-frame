#!/usr/bin/env python3
"""Independent subpixel and affine-photometry matcher tests."""
import unittest
import numpy as np
import cv2
from stereo_photometric_matching import refine_matches


def fixture(gain=1., offset=0., shift=(1.25, -.625)):
    rng = np.random.default_rng(11215)
    raw = rng.integers(40, 201, (256, 256), dtype=np.uint8)
    left = cv2.GaussianBlur(raw, (3, 3), .65)
    shift = np.array(shift, dtype=float)
    moved = cv2.warpAffine(left, np.float32([[1., 0., shift[0]], [0., 1., shift[1]]]), (256, 256))
    right = np.rint(gain * moved.astype(float) + offset).clip(0, 255).astype(np.uint8)
    locations = np.array([[x, y] for y in range(65, 201, 30) for x in range(65, 201, 30)], float)
    seeds = np.column_stack((locations, locations + np.rint(shift)))
    masks = [np.full(left.shape, 255, dtype=np.uint8) for _ in range(2)]
    return [left, right], masks, seeds, shift


class PhotometricMatcherTests(unittest.TestCase):
    def test_no_gain_and_affine_gain(self):
        # Fixed tolerances prior to solver outputs: >=20/25, max axis error
        # .15px, median signed axis bias <.05px.
        for gain, offset in ((1., 0.), (.65, 20.)):
            with self.subTest(gain=gain):
                images, masks, seeds, shift = fixture(gain, offset)
                points = refine_matches(images, masks, seeds)
                self.assertGreaterEqual(len(points), 20)
                error = points[:, 2:] - points[:, :2] - shift
                self.assertLess(np.max(np.abs(error)), .15)
                self.assertLess(np.max(np.abs(np.median(error, axis=0))), .05)

    def test_multiple_subpixel_offsets_and_reversed_views(self):
        for shift in ((-.75, .375), (.125, -.875), (1.5, 1.25), (-1.375, -1.5)):
            with self.subTest(shift=shift):
                images, masks, seeds, expected = fixture(.65, 20., shift)
                forward = refine_matches(images, masks, seeds)
                self.assertGreaterEqual(len(forward), 20)
                error = forward[:, 2:] - forward[:, :2] - expected
                self.assertLess(np.max(np.abs(error)), .15)
                self.assertLess(np.max(np.abs(np.median(error, axis=0))), .05)
                # The second camera as the source is an independently valid
                # reverse problem with its own discretely sampled patch.
                reversed_seeds = np.column_stack((seeds[:, 2:], seeds[:, :2]))
                reverse = refine_matches(images[::-1], masks[::-1], reversed_seeds)
                self.assertGreaterEqual(len(reverse), 20)
                rev_error = reverse[:, 2:] - reverse[:, :2] + expected
                self.assertLess(np.max(np.abs(rev_error)), .15)
                self.assertLess(np.max(np.abs(np.median(rev_error, axis=0))), .05)

    def test_mild_asymmetric_blur_both_directions(self):
        images, masks, seeds, shift = fixture(.65, 20.)
        images[1] = cv2.GaussianBlur(images[1], (3, 3), .8)
        for reverse in (False, True):
            if reverse:
                output = refine_matches(images[::-1], masks[::-1],
                                        np.column_stack((seeds[:, 2:], seeds[:, :2])))
                error = output[:, 2:] - output[:, :2] + shift
            else:
                output = refine_matches(images, masks, seeds)
                error = output[:, 2:] - output[:, :2] - shift
            self.assertGreaterEqual(len(output), 20)
            self.assertLess(np.max(np.abs(error)), .15)
            self.assertLess(np.max(np.abs(np.median(error, axis=0))), .05)

    def test_empty_mask_false_match_and_indices(self):
        images, masks, seeds, _ = fixture()
        out, indices = refine_matches(images, masks, np.empty((0, 4)), return_indices=True)
        self.assertEqual(out.shape, (0, 4))
        self.assertEqual(len(indices), 0)
        x, y = np.rint(seeds[0, 2:]).astype(int)
        masks[1][y, x] = 0
        out, indices = refine_matches(images, masks, seeds, return_indices=True)
        self.assertNotIn(0, indices)
        self.assertEqual(len(out), len(indices))
        np.testing.assert_allclose(out[:, :2], seeds[np.asarray(indices), :2], atol=0., rtol=0.)
        rng = np.random.default_rng(118)
        noise = rng.integers(0, 256, (128, 128), dtype=np.uint8)
        mask = np.full(noise.shape, 255, np.uint8)
        wrong = refine_matches([noise, noise], [mask, mask], [[64., 64., 70., 70.]])
        self.assertEqual(len(wrong), 0)
        flat = np.full((128, 128), 100, dtype=np.uint8)
        self.assertEqual(len(refine_matches([flat, flat], [mask, mask], [[64., 64., 64., 64.]])), 0)


if __name__ == '__main__':
    unittest.main()
