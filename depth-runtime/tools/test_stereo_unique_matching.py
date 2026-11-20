import unittest
import numpy as np
from stereo_unique_matching import refine_matches
from stereo_photometric_matching import refine_matches as roi_matches
from test_stereo_photometric_matching import fixture


class UniqueMatchingTests(unittest.TestCase):
    def test_unique_texture_gain_and_indices(self):
        for gain, offset in [(1., 0.), (.65, 20.)]:
            images, masks, seeds, shift = fixture(gain, offset)
            points, indices = refine_matches(images, masks, seeds, return_indices=True)
            self.assertGreaterEqual(len(points), 20)
            np.testing.assert_array_equal(points[:, :2], seeds[indices, :2])
            self.assertLess(np.max(np.abs(points[:, 2:]-points[:, :2]-shift)), .15)

    def test_periodic_patterns_reject_even_with_exact_identity(self):
        y, x = np.indices((256, 256))
        image = (100+40*np.sin(2*np.pi*x/6)+40*np.cos(2*np.pi*y/6)).astype(np.uint8)
        mask = np.full(image.shape, 255, np.uint8)
        points = np.array([[80., 80., 80., 80.], [120., 120., 120., 120.]])
        self.assertGreater(len(roi_matches([image,image], [mask,mask], points)), 0)
        self.assertEqual(len(refine_matches([image,image], [mask,mask], points)), 0)

    def test_empty_and_mask_rejection(self):
        images, masks, seeds, _ = fixture()
        result, indices = refine_matches(images, masks, [], return_indices=True)
        self.assertEqual(result.shape, (0, 4)); self.assertEqual(len(indices), 0)
        masks[1][:] = 0
        self.assertEqual(len(refine_matches(images, masks, seeds)), 0)


if __name__ == '__main__':
    unittest.main()
